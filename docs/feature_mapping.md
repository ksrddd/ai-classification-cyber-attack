# Feature & Label Mapping

## Label normalization

The 15 raw CICIDS2017 attack labels are collapsed into two schemes by
`src.data.label_mapping`:

### Multiclass (10 classes)

| Raw CICIDS label | Mapped class |
|------------------|--------------|
| `BENIGN` | `BENIGN` |
| `DoS Hulk` | `DoS` |
| `DoS GoldenEye` | `DoS` |
| `DoS slowloris` | `DoS` |
| `DoS Slowhttptest` | `DoS` |
| `DDoS` | `DDoS` |
| `PortScan` | `PortScan` |
| `Bot` | `Bot` |
| `Web Attack \x96 Brute Force` | `Web Attack` |
| `Web Attack \x96 XSS` | `Web Attack` |
| `Web Attack \x96 Sql Injection` | `Web Attack` |
| `FTP-Patator` | `Brute Force` |
| `SSH-Patator` | `Brute Force` |
| `Infiltration` | `Infiltration` |
| `Heartbleed` | `Heartbleed` |
| *(any new label not in this table)* | `Other` |

### Binary

| Raw CICIDS label | Mapped class |
|------------------|--------------|
| `BENIGN` | `Normal` |
| *(every other label)* | `Attack` |

## Normalization rules

`normalize_label` applies, in order:

1. NFKD Unicode normalize.
2. Replace C0/C1 control chars and `U+FFFD` with a single space (this
   strips the `\x96` inside Web Attack labels).
3. Collapse runs of whitespace.
4. Lowercase.
5. Strip leading/trailing whitespace.

The result is an ASCII-only lookup key. New attack variants that follow
existing naming conventions are caught by the prefix-fallback table
(`"dos ..."` -> `DoS`, `"web attack ..."` -> `Web Attack`, etc.) so
adding `DoS GoldenEye` v2 wouldn't break the pipeline.

## Feature columns

CICIDS2017 ships 78 numeric flow-level features plus the `Label`
column. The canonical list lives in `src.data.schema.EXPECTED_FEATURES`
and is asserted on every load via `validate_schema`.

A duplicated column `Fwd Header Length.1` is dropped in cleaning -- it's
a known data-quality bug in the CICIDS distribution.

## Inference schema

`--stage preprocess` writes `data/processed/feature_names.json` -- the
ordered list of feature columns that survived cleaning. The inference
pipeline reads this file and asserts that any uploaded CSV has every
required column (extras are allowed and silently ignored). This is the
single source of truth for "what does the trained model expect to see"
at inference time.
