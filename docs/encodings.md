---
description: Payload of every encoding a stream can name
---

<h1>Encoding Definitions</h1>

---

This page defines the payload of every encoding an MLT stream can name.
Which streams may use which encoding, how the choice is written in the stream header, and where each stream sits are the business of the [v1](specification/v1.md) and [v2](specification/v2.md) specifications.
Those pages define the headers.
This page defines what the bytes after the header mean.

!!! warning "Experimental encodings"
    Encodings marked <span class="experimental"></span> exist only in [MLT v2](specification/v2.md).
    Streams are referred to by their [MLT v2](specification/v2.md#string-columns) names throughout.

## Integer Words

Most of MLT is integers.
Lengths, offsets, ids, dictionary codes, geometry topology and the vertex buffer are all streams of integer words.

An integer stream is decoded in two steps.
The **physical** encoding turns the payload bytes into a sequence of unsigned words.
The **logical** encoding turns those words into the values.
A stream of `Int64`, `UInt64` or `LongId` values has 64-bit words.
Every other integer stream has 32-bit words.

```
values = logical_decode(physical_decode(payload))
```

A difference is signed even when the values are not, so every delta is [zigzag](#zigzag)-encoded.
A plain or run-length-encoded value is ZigZag-encoded only on a stream of a signed type.
Run lengths, dictionary codes, lengths and offsets are never ZigZag-encoded.
Delta arithmetic wraps at the word width.

The steps each logical encoding takes:

--8<-- "diagrams/integer-decode.svg"

## Physical Encodings

### ZigZag {#zigzag}

Used wherever a signed value has to become an unsigned word.
It is not an encoding a stream names; the logical encodings below say where it applies.

The sign is moved to the least significant bit, so that small negative numbers become small positive numbers:

```
zigzag(n)   = (n << 1) ^ (n >> 31)        // arithmetic shift, 32-bit
unzigzag(u) = (u >> 1) ^ -(u & 1)         // logical shift
```

For 64-bit values the shift is `63`.

=== "Mapping"

    --8<-- "diagrams/zigzag.svg"

=== "Values"

    | `n` | `zigzag(n)` |
    |----:|------------:|
    | `0` | `0` |
    | `-1` | `1` |
    | `1` | `2` |
    | `-2` | `3` |
    | `2` | `4` |

### None {#physical-none}

Each word is stored as it is, little-endian, 4 or 8 bytes per word.

In v2 a stream whose logical encoding is also `None` may leave out `byte_length`, since the value count and the word width determine it.
See [byte length](specification/v2.md#byte-length).

### VarInt {#varint}

Each word is stored in 7-bit groups, least significant group first.
Bit 7 of each byte is set when another byte follows.
A 32-bit word takes 1 to 5 bytes and a 64-bit word 1 to 10.

[View example](inspector/app/?fixture=0x02%2Fprops_u32_np.mlt&at=column%5B0%5D){target=_blank .inspector-example} - four `9000`s, two bytes each, the first byte of each with bit 7 set.

=== "Bits"

    --8<-- "diagrams/varint.svg"

=== "Values"

    ```
    300 = 0b0000_0001_0010_1100
        -> groups (LSB first): 0101100, 0000010
        -> bytes:              0xAC,    0x02
    ```

This is the unsigned varint of [Protocol Buffers](https://protobuf.dev/programming-guides/encoding/#varints) and the length prefix every MLT header uses.
Signed values go through [zigzag](#zigzag) first, where the logical encoding says so.

### FastPFOR {#fastpfor}

A block codec that stores each block of words at the bit width most of them need, and patches the few words that need more from separate exception arrays.
A handful of outliers does not raise the width of the whole block.
It is described by [Lemire and Boytsov, *Decoding billions of integers per second through vectorization*](https://arxiv.org/pdf/1209.2137.pdf), and MLT stores the layout of the `FastPFOR` and `VariableByte` composite of its reference libraries, with 256-value blocks.
v1 and v2 use the same layout, for 32-bit values only.
A stream of 64-bit words cannot use FastPFOR.

[View example](inspector/app/?fixture=0x02%2Fids_fpf.mlt&at=column%5B0%5D){target=_blank .inspector-example} - an id stream in FastPFOR.

The payload consists of whole 32-bit words, each stored big-endian:

```
payload := [u32 block_values]          values in whole blocks, a multiple of 256
           [page; ceil(block_values / 65536)]
           [u8 tail[...]]              the remaining values, variable-byte encoded
```

`block_values` is the value count rounded down to a multiple of `256`.
The values after the last whole block, at most `255`, are the [tail](#fastpfor-tail).
A stream of fewer than `256` values has `block_values = 0`, no page, and only a tail.
A stream of zero values has an empty payload.
A payload whose length is not a multiple of 4 MUST be rejected.

The block metadata and the tail are bytes, packed four to a word with the first byte in the low 8 bits.
Since the word is then stored big-endian, each group of four of those bytes appears reversed in the payload.

--8<-- "diagrams/fastpfor-layout.svg"

#### Pages

A page holds up to `65536` values, which is `256` blocks.
Every page but the last is full.
Exceptions are collected per page, so that the blocks of a page share one set of exception arrays.

```
page := [u32 meta_offset]              words from this word to meta_size, 1 + the packed words
        [packed block; blocks]         8 * width words each
        [u32 meta_size]                bytes of block metadata
        [u8 meta[meta_size]]           zero-padded to whole words
        [u32 bitmap]
        [exception array; popcount(bitmap)]
```

The metadata is one entry per block, in block order:

```
block_meta := [u8 width]               0-32
              [u8 exceptions]          0-255
              if exceptions > 0:
                [u8 max_width]         bit width of the widest value in the block
                [u8 position; exceptions]
```

`width` is the bit width the block is packed at.
A value that needs more is an **exception**: its low `width` bits stay in the block, and its high bits go to an exception array.
`position` is the index of the exception in the block, `0` to `255`, and encoders write the positions in ascending order.

An exception array can exist for each width `k` from `2` to `32`, and bit `k - 1` of the bitmap says whether it does.
Array `k` holds the high bits of every exception in the page with `max_width - width = k`, in block order and then position order.

```
exception_array := [u32 count]
                   [count values at k bits each]
```

The values of an array are [packed](#fastpfor-packing) in groups of 32.
The last group is padded with zero values, and the words the padding alone would fill are not written, so an array takes `ceil(count * k / 32)` words.
Arrays follow in ascending `k`.

A block is decoded by unpacking its `256` values at `width` bits, then patching its exceptions:

```
index = max_width - width
for each position p:
    index == 1:  value[p] |= 1 << width
    otherwise:   value[p] |= next value of array[index] << width
```

Arrays are split by `k` so that each packs at its exact width and no exception stores its own width.
An exception that is a single bit wider than the block has no array: its high part can only be `1`.
Every array has its own read position, which starts at its first value on each page.

A decoder MUST reject
- a `width` or `max_width` above `32`,
- a `max_width` that is not above `width` where there are exceptions,
- an exception without a value left in its array,
- an array with more than `65536` values,
- and a payload that ends inside a page.

#### Packing {#fastpfor-packing}

32 values at `k` bits each are one continuous LSB-first bit stream of `k` words: value `j` occupies bits `j * k` to `j * k + k - 1`, with bit `i` of the stream being bit `i mod 32` of word `i / 32`.
A block is 8 such groups, and an exception array as many as it needs.
A `width` of `0` takes no words, and every value is `0`.

--8<-- "diagrams/fastpfor-sequential.svg"

#### Tail {#fastpfor-tail}

The tail is variable-byte encoded, with no count and no header.
Each value is stored in 7-bit groups, least significant group first, and bit 7 of its **last** byte is set.
This is the reverse of [varint](#varint), where bit 7 marks a byte that is followed by another.

A value takes 1 to 5 bytes, and the fifth carries the top 4 bits.
The bytes are padded with `0x00` to a multiple of 4.
A padding byte has bit 7 clear, so it never ends a value, and a decoder discards the unfinished value at the end.
A decoder reads until the payload ends, and the number of values it read MUST be the stream's value count minus `block_values`.

#### Choosing the width

The width is the encoder's choice, and any width that decodes to the values is valid.
The reference encoder counts the values of a block by bit length and tries each `width` below the widest value's `max_width`.
With `e` values wider than `width`, it costs

```
256 * width + e * 8 + e * (max_width - width) + 8        bits
```

for the packed block, the position bytes, the exception bits and the `max_width` byte.
It picks the cheapest, and `max_width` itself when no smaller width beats `256 * max_width`.
It never picks a `width` that makes every value an exception.

#### Example

Take 258 values, `i mod 8` for `i` from `0`, except `300` at `5`, `20` at `77`, `5` at `256` and `200` at `257`.
The first 256 fit in `3` bits except two, so `width = 3` and `max_width = 9`.
The last two are the tail.
The payload is 33 words, as stored:

```
00 00 01 00                 block_values   256
00 00 00 19                 meta_offset    25: 1 + 24 packed words
88 FA 46 88 ... FA C6 88 FA packed block   24 words, 256 values at 3 bits
00 00 00 05                 meta_size      5 bytes
05 09 02 03                 meta           bytes 03 02 09 05: width 3, 2 exceptions, max_width 9, at 5
00 00 00 4D                                bytes 4D 00 00 00: and at 77, padded
00 00 00 20                 bitmap         bit 5: k = 9 - 3 = 6
00 00 00 02                 count          2
00 00 00 A5                 array 6        37 and 2 at 6 bits each: 300 >> 3 and 20 >> 3
00 81 48 85                 tail           bytes 85 48 81 00: 5, then 200, padded
```

## Logical Encodings

The logical encoding is applied to the words the physical step produced.

### None {#logical-none}

The words are the values.
On a signed stream each word is [zigzag](#zigzag)-decoded.

### Delta {#delta}

Each word is the difference from the previous value.
The first value's predecessor is `0`.
Each difference is [zigzag](#zigzag)-encoded before it becomes a word, whatever the stream's type.

[View example](inspector/app/?fixture=0x02%2Fids_opt_delta.mlt&at=column%5B0%5D){target=_blank .inspector-example} - ids `100`, `101`, `105`, `106`, stored as the steps between them.

```
delta[0] = values[0] - 0
delta[i] = values[i] - values[i - 1]

decode: values[i] = values[i - 1] + delta[i]
```

Example:

```
values: [100, 105, 102]
deltas: [100, 5, -3]
words:  [200, 10, 5]           zigzag
```

Delta suits monotonic sequences such as ids and offsets, whose differences are small however large the values are.

### RLE {#rle}

The values are stored as runs, each a `(run_length, value)` pair that expands to `run_length` copies of `value`.
On a signed stream `value` is [zigzag](#zigzag)-encoded.
Run lengths never are.

[View example](inspector/app/?fixture=0x02%2Fmvalues_rle.mlt&at=m_value%5B0%5D){target=_blank .inspector-example} - five `5`s and three `7`s, as two runs.

The two tile versions lay the runs out differently:

| | Payload | Run count |
|---|---|---|
| **v1** | All run lengths, then all values, physically encoded as one word sequence | `runs` varint in the stream header |
| **v2** | Interleaved `(run_length, value)` pairs as varints, no physical field | Not stored; read pairs until `byte_length` is exhausted |

For the input `[5, 5, 5, 3, 3, 3, 3]`:

=== "Runs"

    --8<-- "diagrams/rle.svg"

=== "Values"

    ```
    runs:   [3, 4]
    values: [5, 3]

    v1 words: [3, 4, 5, 3]         header: runs = 2, num_rle_values = 7
    v2 bytes: 03 05 04 03          header: 7 values from context
    ```

In both versions the decoded element count is known before the payload is read.
v1 stores it in the header as `num_rle_values`.
v2 takes it from the stream's value count.
Run lengths that do not sum to exactly that count MUST be rejected.
A v2 payload with an odd number of varints MUST be rejected.

### Delta-RLE {#delta-rle}

[Delta](#delta) followed by [RLE](#rle).
The deltas are ZigZag-encoded, and those unsigned words are run-length encoded.
Decoding undoes both steps in reverse order: expand the runs, then undo ZigZag and prefix-sum.

[View example](inspector/app/?fixture=0x02%2Fids_delta_rle.mlt&at=column%5B0%5D){target=_blank .inspector-example} - four equal ids: one step of `103`, then a run of three `0`s.

=== "Runs"

    --8<-- "diagrams/delta-rle.svg"

=== "Values"

    ```
    values: [10, 11, 12, 13, 20, 20, 20]
    deltas: [10, 1, 1, 1, 7, 0, 0]
    words:  [20, 2, 2, 2, 14, 0, 0]          zigzag
    runs:   (1, 20) (3, 2) (1, 14) (2, 0)
    ```

Delta-RLE suits sequences with a constant step, such as `[1, 2, 3, ...]`, which become one run.

## Boolean Streams

Boolean columns and v1 `Present` streams hold one bit per value.

### Bitmap {#bitmap}

Bit `i % 8` of byte `i / 8` is value `i`, LSB-first.
The bitmap is `ceil(count / 8)` bytes.
Bits past `count` in the final byte are padding and MUST be ignored.

[View example](inspector/app/?fixture=0x02%2Fprop_i32_val_null.mlt&at=present){target=_blank .inspector-example} - a value, then a null: bits `1` and `0`.

=== "Bits"

    --8<-- "diagrams/bitmap.svg"

=== "Values"

    ```
    values: [1, 0, 1, 1, 0, 1]
    byte:   0b0010_1101 = 0x2D
    ```

### Boolean RLE {#boolean-rle}

v1 compresses the [bitmap](#bitmap) with the byte-level run-length encoding of [ORC](https://orc.apache.org/specification/ORCv1/#byte-run-length-encoding).
v2 keeps it as the `ByteRLE` [presence encoding](specification/v2.md#presence-encodings), an extension for tiles read without transport compression.
The payload is a sequence of runs, each a control byte followed by the bytes it describes.

[View example](inspector/app/?fixture=0x01%2Fids_opt.mlt&at=present){target=_blank .inspector-example} - five ids, one of them missing, as byte runs.

| Control byte `c` | Meaning |
|---|---|
| `0`-`127` | A repeated run: the next byte, `c + 3` times |
| `128`-`255` | A literal run: the next `256 - c` bytes as they are |

A repeated run is 3 to 130 bytes and a literal run 1 to 128.
The runs expand to exactly `ceil(count / 8)` bitmap bytes.
A payload that expands to more, or ends before that, MUST be rejected.

=== "Runs"

    --8<-- "diagrams/boolean-rle.svg"

=== "Values"

    ```
    bitmap:  FF FF FF FF FF 2D
    payload: 02 FF FF 2D
             02 FF          repeated run: 2 + 3 = 5 copies of FF
             FF 2D          literal run: 256 - 255 = 1 byte, 2D
    ```

The v1 stream header carries no `runs` or `num_rle_values` for a boolean stream.
Both follow from `num_values`.

## Float Streams

### Plain Floats {#plain-floats}

IEEE 754 words, little-endian, 4 bytes for a `Float` and 8 for a `Double`.
This is the only float encoding v1 and v2 have.

[View example](inspector/app/?fixture=0x02%2Fprop_f64_max_np.mlt&at=column%5B0%5D){target=_blank .inspector-example} - raw IEEE 754 words.

## Strings {#strings}

String values, dictionary values and FSST symbol tables are byte blobs.
A blob's count is its `byte_length`, and a lengths stream beside it says where each value ends.

A string column is stored with one of four encodings, which combine a blob with an optional dictionary:

| Encoding | Stores |
|---|---|
| Plain | Every value's bytes as [plain bytes](#plain-bytes) |
| Dictionary | The distinct values once, and a code per value |
| FSST | Every value's bytes, compressed with [FSST](#fsst) |
| FSST dictionary | The distinct values once, compressed with [FSST](#fsst), and a code per value |

The streams of each encoding and how a column announces it are the business of [v1](specification/v1.md#string-columns) and [v2](specification/v2.md#string-columns).

### Plain Bytes {#plain-bytes}

The bytes as they are.
Value `i` is the `lengths[i]` bytes that follow the first `lengths[0] + ... + lengths[i - 1]`.
Each value MUST be valid UTF-8.

[View example](inspector/app/?fixture=0x02%2Fprops_str_plain_np.mlt&at=column%5B0%5D){target=_blank .inspector-example} - several strings, each its length's worth of bytes in turn.

### Front Coding <span class="experimental"></span> {#front-coding}

Neighboring entries of a sorted dictionary often share a prefix, for example `Main Street`, `Main Street North` and `Maple Avenue`.
Front coding stores the length of the prefix shared with the previous entry, and only the suffix bytes.

[View example](inspector/app/?fixture=0x02%2Fprops_str_front_dict_np.mlt&at=column%5B0%5D){target=_blank .inspector-example} - a dictionary whose entries share prefixes.

The lengths stream that precedes the blob holds `2N` values: `N` shared-prefix lengths, then `N` suffix lengths.
The blob holds the `N` suffixes back to back.

```
entry[i] = entry[i - 1][..prefix_lengths[i]] ++ suffix[i]
```

The first entry's prefix length is always `0`.
A prefix length longer than the previous entry, a lengths stream with an odd count, or suffix bytes left over after the last entry MUST be rejected.

Entries are reconstructed sequentially.
Random access requires a scan from the start of the dictionary.
Prefix lengths are in bytes and may split a multi-byte character.
Only the reconstructed entry has to be valid UTF-8.

=== "Prefixes"

    --8<-- "diagrams/front-coding.svg"

=== "Values"

    ```
    entries:  ["Main Street", "Main Street North", "Maple Avenue"]
    prefixes: [0, 11, 2]
    suffixes: [11, 6, 10]
    lengths:  [0, 11, 2, 11, 6, 10]
    blob:     "Main Street" " North" "ple Avenue"
    ```

Front coding can be combined with [FSST](#fsst).
The corpus is front-coded first and then FSST-compressed.
An encoder SHOULD rank the combinations as [choosing encodings](specification/v2.md#choosing-encodings) says.

### FSST {#fsst}

Fast Static Symbol Table compression replaces frequent byte sequences of up to 8 bytes with one-byte codes.
Unlike a dictionary, it compresses strings that merely share substrings, such as localized country names.
It decodes quickly, since every code is one byte.

[View example](inspector/app/?fixture=0x02%2Fprops_str_fsst_dict_np.mlt&at=column%5B0%5D){target=_blank .inspector-example} - a symbol table and the codes into it.

An FSST-compressed blob comes with two streams that carry its symbol table:

| Stream | Holds |
|---|---|
| `SymbolLengths` | The byte length of each symbol, an integer stream |
| `SymbolTable` | The symbols, back to back, a plain byte blob |

There are at most `255` symbols, numbered `0` to `254` in the order the two streams list them.

The `Corpus` blob is a sequence of codes:

| Byte `b` | Meaning |
|---|---|
| `0`-`254` | Symbol `b`, expanded to its bytes |
| `255` | Escape: the next byte is output as it is |

A code naming a symbol the table does not have, or a corpus ending on an escape byte, MUST be rejected.

The corpus compresses all values as one buffer, so a symbol may span two values.
The lengths stream beside the corpus holds the **uncompressed** length of each value.
A decoder expands the whole corpus and then splits it by those lengths.

=== "Expansion"

    --8<-- "diagrams/fsst.svg"

=== "Values"

    ```
    symbols:      ["ab", "cd"]           SymbolLengths = [2, 2], SymbolTable = "abcd"
    values:       ["abcd", "abz"]        Lengths = [4, 3]
    corpus bytes: 00 01 00 FF 7A         ab cd ab <esc> z
    ```

The algorithm that trains the table is described by [Boncz, Neumann and Leis, *FSST: Fast Random Access String Compression*](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf).
Different implementations train different tables for the same input.
A decoder reads the table from the stream, so this only matters when comparing the output of two encoders.

### Dictionary {#string-dictionary}

The distinct values are stored once, and a code names one of them per element.
Repeated values such as a city or a country cost one code instead of their bytes.

[View example](inspector/app/?fixture=0x02%2Fprops_str_dict_shuffled_np.mlt&at=codes){target=_blank .inspector-example} - three distinct strings and one code per feature.

The codes are an unsigned integer stream, so any [logical](#logical-encodings) and [physical](#physical-encodings) encoding can compress them.
The dictionary is a byte blob, [plain](#plain-bytes), [front-coded](#front-coding) or [FSST](#fsst)-compressed.
Its lengths are those of the distinct values.

```
values: ["Oslo", "Bergen", "Oslo", "Oslo"]
codes:  [0, 1, 0, 0]
dict:   ["Oslo", "Bergen"]
```

A code at or past the dictionary's count MUST be rejected.
A dictionary beats plain strings once values repeat, and loses when they are mostly distinct, so an encoder SHOULD compare the stored size of both.

### Shared Dictionary {#shared-dictionary}

Several string columns, such as `name:en`, `name:de` and `name:fr`, index into one dictionary.
The dictionary is written once, and each member column then stores only its presence and its codes.

[View example](inspector/app/?fixture=0x02%2Fprops_shared_dict_codes.mlt&at=column%5B0%5D){target=_blank .inspector-example} - several columns reading one corpus.

The dictionary can be plain, front-coded or FSST-compressed, as for a single column.
See [v1](specification/v1.md#shared-dictionary-columns) and [v2](specification/v2.md#shared-dictionary-columns) for the group header each version writes.

## Vertex Streams

The vertex buffer holds $x$ and $y$ interleaved: $[x_0, y_0, x_1, y_1, \ldots]$.
Its words are 32-bit and signed.

### Componentwise Delta {#componentwise-delta}

Each coordinate is a delta to the same coordinate of the previous vertex.
`x` and `y` keep separate predecessors, both starting at `0`.
Each delta is [zigzag](#zigzag)-encoded.

[View example](inspector/app/?fixture=0x02%2Fline.mlt&at=vertices){target=_blank .inspector-example} - three vertices, each stored as its step from the one before.

```
dx[i] = x[i] - x[i - 1]        x[-1] = 0
dy[i] = y[i] - y[i - 1]        y[-1] = 0
words = [zigzag(dx[0]), zigzag(dy[0]), zigzag(dx[1]), zigzag(dy[1]), ...]
```

```
vertices: (100, 200), (105, 210), (102, 215)
deltas:   (100, 200), (5, 10), (-3, 5)
words:    [200, 400, 10, 20, 5, 10]
```

The v2 `Vertex` family also has a plain `Delta`, which is the integer [delta](#delta) over the flat word sequence and does not separate the components.

### Vertex Dictionary {#vertex-dictionary}

The distinct vertices are stored once in a `VertexDict` stream, and a `VertexOffsets` stream holds one index into them per vertex.
`VertexOffsets` is an ordinary unsigned integer stream.
`VertexDict` is a vertex stream and carries any of the encodings in this section.

[View example](inspector/app/?fixture=0x02%2Fpoint_morton_dictionary.mlt&at=vertex_dict){target=_blank .inspector-example} - vertices stored once and indexed.

=== "Offsets"

    --8<-- "diagrams/vertex-dictionary-offsets.svg"

=== "Values"

    ```
    VertexOffsets: [0, 1, 2, 1, 0, 2]
    VertexDict:    [(0,0), (10,10), (20,20)]
    vertices:      (0,0), (10,10), (20,20), (10,10), (0,0), (20,20)
    ```

An offset at or past the dictionary's vertex count MUST be rejected.

#### Hilbert Order

Encoders SHOULD sort the dictionary along a Hilbert curve, so that the deltas between neighboring entries stay short.
The order is not part of the format.
A decoder resolves vertices through `VertexOffsets` and never depends on the order.

The reference encoders use the curve of the [`hilbert_2d`](https://crates.io/crates/hilbert_2d) crate's `Hilbert` variant on a $2^{\mathit{bits}} \times 2^{\mathit{bits}}$ grid.
`shift` and `bits` are derived as for [Morton](#morton) below, and each shifted coordinate is masked to 16 bits before it is placed on the grid.
Two vertices with the same curve key are one dictionary entry.

### Morton {#morton}

A Morton, or Z-order, code interleaves the bits of a coordinate pair into one integer.
Nearby vertices get nearby codes, so a sorted Morton dictionary has small deltas.
Only a `VertexDict` stream uses it.

[View example](inspector/app/?fixture=0x02%2Fpoint_morton_dictionary.mlt&at=vertex_dict){target=_blank .inspector-example} - vertices as Morton codes, delta-encoded.

The parameters are two varints in the stream header:

| Parameter | Meaning |
|---|---|
| `bits` | Bits per axis, at most `16` |
| `shift` | Added to `x` and to `y` before interleaving, so that both are non-negative |

`bits > 16` MUST be rejected.

Encoders derive both from the whole layer's vertices.
`shift` is `-min` when the smallest coordinate `min` on either axis is negative, else `0`.
`bits` is the bit width of `max + shift`, the largest shifted coordinate.

Encoding takes two steps.
Shift both coordinates, then interleave them: bit `i` of `sx` goes to bit `2i` of the code, and bit `i` of `sy` to bit `2i + 1`.

```rust
fn encode(x: i32, y: i32, bits: u32, shift: u32) -> u32 {
    let sx = (i64::from(x) + i64::from(shift)) as u32; // MUST be in 0..2^bits
    let sy = (i64::from(y) + i64::from(shift)) as u32;
    let mut code = 0;
    for i in 0..bits {
        code |= ((sx >> i) & 1) << (2 * i);
        code |= ((sy >> i) & 1) << (2 * i + 1);
    }
    code
}
```

Decoding walks the same bits back and undoes the shift.

```rust
fn decode(code: u32, bits: u32, shift: u32) -> (i32, i32) {
    let mut x = 0;
    let mut y = 0;
    for i in 0..bits {
        let mask = 1 << (2 * i);
        x |= (code & mask) >> i;
        y |= ((code >> 1) & mask) >> i;
    }
    (x.wrapping_sub(shift) as i32, y.wrapping_sub(shift) as i32)
}
```

=== "Bits"

    --8<-- "diagrams/morton.svg"

=== "Values"

    ```
    (x, y) = (5, 3), shift = 0, bits = 3
    sx = 0b101, sy = 0b011
    code bits, from bit 0: x0 y0 x1 y1 x2 y2 = 1 1 0 1 1 0
    code = 0b011011 = 27
    ```

The loops are the definition.
The reference Rust codec spreads all 16 bits of an axis at once with masks, which gives the same code for any `bits` up to `16`:

```rust
/// Bit `i` of `v` lands on bit `2i`.
fn spread(mut v: u32) -> u32 {
    v &= 0xFFFF;
    v = (v | (v << 8)) & 0x00FF_00FF;
    v = (v | (v << 4)) & 0x0F0F_0F0F;
    v = (v | (v << 2)) & 0x3333_3333;
    v = (v | (v << 1)) & 0x5555_5555;
    v
}

/// Bit `2i` of `v` lands on bit `i`.
fn compact(mut v: u32) -> u32 {
    v &= 0x5555_5555;
    v = (v | (v >> 1)) & 0x3333_3333;
    v = (v | (v >> 2)) & 0x0F0F_0F0F;
    v = (v | (v >> 4)) & 0x00FF_00FF;
    v = (v | (v >> 8)) & 0x0000_FFFF;
    v
}

let code = spread(sx) | (spread(sy) << 1);
let sx = compact(code);
let sy = compact(code >> 1);
```

The codes are stored as an integer stream of unsigned 32-bit words.
The stream holds the first code, then each code's difference from the previous one.
v1 names this `MortonDelta`; v2 names it `Morton` in the `Vertex` family.
There is no plain or RLE Morton encoding.

The deltas are plain differences, not ZigZag-encoded.
The dictionary is sorted ascending by code, so every difference is non-negative.

```
dict codes: [27, 30, 45]
words:      [27, 3, 15]
```

### Predicted <span class="experimental"></span> {#predicted}

Predicted vertices store each vertex of a plain layout's vertex stream as what a guess from the vertices decoded before it misses, or as a copy of vertices decoded before it.
A `VertexDict` stream MUST NOT use it, and neither may an [`(x, y, z)`](specification/v2.md#z-coordinates) stream.

It does no entropy coding of its own.
Its residuals are bytes, grouped by the kind of vertex they belong to, so that the gzip, brotli or zstd a tile is served with sees homogeneous byte-aligned symbols.
Every step is integer arithmetic, so every platform decodes the same bits.

The guesses follow the shape of map geometry:

- a ring or line starts near where the previous ring or line started, or, with starts from ends, where the previous run ended, which fits road networks stored as chained segments,
- the fourth vertex of a four-vertex ring completes a parallelogram,
- every other vertex is near the one before it.

Neighbouring polygons share borders and roads share junctions, so a stretch of vertices an earlier ring or line already holds is stored as a copy of it, forwards or backwards.

#### Runs and Classes {#predicted-runs}

A run is the vertex range of each ring, else of each part, else of each geometry, taken from the deepest offset level the geometry section has.
Each run takes its kind from the geometry type of the feature that owns it:

| Feature type | Run kind |
|---|---|
| `Point`, `MultiPoint` | Point |
| `LineString`, `MultiLineString` | Line |
| `Polygon`, `MultiPolygon` | Ring |

A layer with none of the three offset levels has one Point run per vertex, and every feature MUST then be a `Point` or `MultiPoint` with one vertex each.
The runs MUST cover exactly the stream's `num_values` vertices, and every run MUST be owned by a feature.
Because the runs come from the topology, the stream is decoded after the topology streams.

Vertex `i` of a run of length `len` belongs to one of six classes:

| Code | Class | Vertices | Guess |
|---:|---|---|---|
| `0` | Start | `i = 0` of a Line or Ring run | the first vertex of the previous Line or Ring run, `(0, 0)` before the first; with starts from ends, the last vertex of the previous run of any kind, `(0, 0)` before the first |
| `1` | Point | every vertex of a Point run | the vertex before it in the stream, `(0, 0)` before the first |
| `2` | Ring | `i ≥ 1` of a Ring run, except a Corner | the vertex before it |
| `3` | Corner | `i = 3` of a Ring run with `len = 4` | `v[0] + v[2] - v[1]`, the parallelogram |
| `4` | Line1 | `i = 1` of a Line run | the vertex before it |
| `5` | Line | `i ≥ 2` of a Line run | the vertex before it |

The residual is the vertex minus its guess.
Coordinates and residuals are computed in at least 64-bit signed integers, and a decoded coordinate outside the 32-bit signed range MUST be rejected.

#### Copies {#predicted-copies}

A copy writes `n ≥ 2` vertices of a Line or Ring run by reading earlier vertices of one source run, starting `distance` vertices back from the vertex being written.
It reads forwards or backwards.
On a closed source ring that is already complete, reading wraps from its last vertex to its first, or backwards from its first to its last.
A copy never leaves its source run, never reads a vertex not yet written, and never extends past the end of the run being written.
Point runs never copy.

#### Payload {#predicted-payload}

```
payload := [u8 flags]
           [varint ops_length] [ops]        only when flags bit 0 is set
           residual_block * 6               classes 0 to 5, in code order

flags          bit 0: an ops block follows, bit 2: starts from ends,
               bit 1 and bits 3-7 are reserved and MUST be 0
ops            per Line or Ring run, in stream order, also for a run of length 0:
                 [varint copy_count]
                 copy_count * ([varint gap] [varint (distance << 1) | backwards] [varint n - 2])
residual_block := [varint block_length] [varint low_length] [u8 low[low_length]] [high]
```

`gap` is the number of vertices the run guesses before this copy, counted from the run's start or the end of its previous copy.
`block_length` counts the `low_length` varint, the low plane and the high plane.

A class's residuals are its `(x, y)` residual pairs in vertex order, `x` first, each [ZigZag](#zigzag)-encoded as a 64-bit value `z`.
Each `z` below `255` is one byte of the low plane.
A larger `z` is the byte `255` in the low plane and the canonical varint of `z - 255` in the high plane, in the same order.

A decoder MUST reject reserved flag bits, a copy that breaks the rules above, a class whose residuals run out or are left over, ops or high-plane bytes left over, and bytes after the last block.

```
prev = prev_start = (0, 0)
for each run (len, kind):
    copies = read copy_count, if bit 0 is set and kind is not Point
    for i in 0..len:
        if a copy starts at i: write its n vertices, i += n, continue
        c = class of vertex i
        r = next residual pair of class c
        v[i] = prev + r                         Point, i = 0
             = prev_start + r                   Start, bit 2 clear
             = prev + r                         Start, bit 2 set
             = v[0] + v[2] - v[1] + r           Corner
             = v[i-1] + r                       otherwise
    if len > 0:
        prev_start = v[0], unless kind is Point
        prev = v[len - 1]
```

An encoder SHOULD compare predicted vertices with componentwise delta and the vertex dictionary by the size they take after the transport compression the tile is served with.
It SHOULD try both start guesses the same way.
Starts from ends pay off when each line begins where the one before it ended, as in a road network stored as chained segments.

!!! example

    A square ring `(10, 10), (20, 10), (20, 20), (10, 20)`, then a line `(0, 0), (5, 0), (10, 1)`:

    | Vertex | Class | Guess | Residual | ZigZag |
    |---|---|---|---|---|
    | `(10, 10)` | Start | `(0, 0)` | `(10, 10)` | `20, 20` |
    | `(20, 10)` | Ring | `(10, 10)` | `(10, 0)` | `20, 0` |
    | `(20, 20)` | Ring | `(20, 10)` | `(0, 10)` | `0, 20` |
    | `(10, 20)` | Corner | `(10, 10) + (20, 20) - (20, 10)` | `(0, 0)` | `0, 0` |
    | `(0, 0)` | Start | `(10, 10)` | `(-10, -10)` | `19, 19` |
    | `(5, 0)` | Line1 | `(0, 0)` | `(5, 0)` | `10, 0` |
    | `(10, 1)` | Line | `(5, 0)` | `(5, 1)` | `10, 2` |

    ```
    flags:   00                   no copies, starts from starts
    Start:   05 04 14 14 13 13
    Point:   01 00
    Ring:    05 04 14 00 00 14
    Corner:  03 02 00 00
    Line1:   03 02 0A 00
    Line:    03 02 0A 02
    ```
