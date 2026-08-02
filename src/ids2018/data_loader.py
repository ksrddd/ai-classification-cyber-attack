"""Memory-bounded stratified sampler for CSE-CIC-IDS2018.

The corpus is ~13M rows / 6.7 GB spread over ten daily CSVs, which will not
fit in RAM on a normal workstation. Loading everything just to throw 97.7%
of it away is wasteful, so this module uses a **two-pass streaming scan**:

    Pass 1 (cheap)  Read only the ``Label`` column of every file in chunks.
                    Result: one int16 code per row of the whole corpus
                    (~26 MB for 13M rows), plus a per-file row offset table.

    Plan            Run ``StratifiedShuffleSplit`` over that label vector to
                    choose exactly ``SAMPLE_SIZE`` row positions. This is
                    genuine stratified sampling -- never ``df.sample()`` --
                    so every one of the 15 classes is represented at its
                    original proportion.

    Pass 2 (heavy)  Re-read the same files in identically-sized chunks and
                    keep only the chosen row positions. Peak memory is one
                    chunk plus the growing 300k-row result.

Positional alignment between the two passes is the correctness-critical
invariant, so pass 2 re-checks the label of every selected row against what
pass 1 recorded and raises if they ever disagree.

Two quirks of this specific dataset are handled here:

* ``02-20-2018.csv`` was exported with four extra identifier columns
  (Flow ID / Src IP / Src Port / Dst IP) that the other nine files lack.
* Every file contains repeated CSV header lines concatenated mid-file.
  These are *not* dropped during reading -- doing so would shift row
  positions between passes. They are instead marked invalid (code -1) and
  simply never sampled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.ids2018.config import (
    CHUNKSIZE,
    EMBEDDED_HEADER_TOKEN,
    LABEL_ALIASES,
    LABEL_COL,
    MIN_PER_CLASS,
    RANDOM_STATE,
    SAMPLE_SIZE,
    TEXT_IDENTITY_COLS,
    TIMESTAMP_COL,
)

logger = logging.getLogger(__name__)

# Shared across both passes. Any difference in these options between the
# two passes could change how many rows pandas yields and silently break
# positional alignment, so they live in one place.
_READ_OPTS = {
    "engine": "c",
    "skip_blank_lines": False,  # a blank line must still consume a position
    "on_bad_lines": "skip",
}


# ----------------------------------------------------------------------
# Label normalisation
# ----------------------------------------------------------------------
def normalise_labels(raw: pd.Series) -> pd.Series:
    """Map raw label strings to their canonical spelling.

    Unknown labels are left as-is (title-cased) rather than dropped, so a
    future re-release of the dataset with a new attack type shows up in the
    class distribution instead of vanishing silently.
    """
    stripped = raw.astype("string").str.strip()
    key = stripped.str.lower()
    mapped = key.map(LABEL_ALIASES)
    # Fall back to the original spelling where no alias matched.
    return mapped.fillna(stripped)


@dataclass
class FileIndex:
    """Pass-1 result for a single CSV."""

    path: Path
    n_rows: int  # rows pandas yielded, including invalid ones
    codes: np.ndarray  # int16 per row; -1 marks an unusable row


@dataclass
class CorpusIndex:
    """Pass-1 result for the whole corpus."""

    files: list[FileIndex]
    classes: np.ndarray  # str array; position == integer code
    offsets: np.ndarray  # int64, len = n_files + 1; global row offsets

    @property
    def codes(self) -> np.ndarray:
        """Concatenated per-row label codes across all files."""
        return np.concatenate([f.codes for f in self.files])

    @property
    def n_rows(self) -> int:
        return int(self.offsets[-1])


# ----------------------------------------------------------------------
# Index cache
# ----------------------------------------------------------------------
def _fingerprint(files: list[Path]) -> np.ndarray:
    """Identify the raw corpus by (name, size, mtime) of every file.

    Cheap to compute and enough to notice a re-download or an added day,
    both of which would invalidate every cached row position.
    """
    return np.array(
        [f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}" for p in files], dtype=str
    )


def save_index(index: CorpusIndex, path: Path) -> None:
    """Persist the pass-1 result so later runs can skip the CSV scan."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        codes=index.codes,
        offsets=index.offsets,
        classes=index.classes,
        paths=np.array([str(f.path) for f in index.files], dtype=str),
        fingerprint=_fingerprint([f.path for f in index.files]),
    )
    logger.info("Label index cached to %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_index(path: Path, files: list[Path]) -> CorpusIndex | None:
    """Load a cached pass-1 result, or ``None`` if it does not match ``files``."""
    if not path.exists():
        return None

    with np.load(path, allow_pickle=False) as data:
        cached_paths = [Path(p) for p in data["paths"]]
        if cached_paths != files or not np.array_equal(data["fingerprint"], _fingerprint(files)):
            logger.warning("Cached label index does not match the raw CSVs -- rescanning")
            return None

        offsets = data["offsets"]
        codes = data["codes"]
        index = CorpusIndex(
            files=[
                FileIndex(
                    path=path_i,
                    n_rows=int(offsets[i + 1] - offsets[i]),
                    codes=codes[offsets[i] : offsets[i + 1]],
                )
                for i, path_i in enumerate(cached_paths)
            ],
            classes=data["classes"],
            offsets=offsets,
        )

    logger.info(
        "Reusing cached label index: %s rows, %d classes (CSV scan skipped)",
        f"{index.n_rows:,}",
        len(index.classes),
    )
    return index


# ----------------------------------------------------------------------
# Pass 1 -- label-only scan
# ----------------------------------------------------------------------
def _encode_chunk(labels: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Turn one chunk of raw label strings into int16 codes.

    ``vocab`` is grown in place as new labels are discovered, so codes are
    assigned in first-seen order here and remapped to alphabetical order
    once the whole corpus has been scanned. Encoding per chunk -- rather
    than holding 13M Python strings until the end -- is what keeps this
    pass to tens of megabytes instead of gigabytes.
    """
    canon = normalise_labels(labels)
    invalid = (canon.isna() | (canon.str.lower() == EMBEDDED_HEADER_TOKEN)).fillna(True).to_numpy()

    codes = np.full(len(canon), -1, dtype=np.int16)
    values = canon.to_numpy(dtype=object)
    for pos in np.flatnonzero(~invalid):
        name = values[pos]
        code = vocab.get(name)
        if code is None:
            code = vocab[name] = len(vocab)
        codes[pos] = code
    return codes


def scan_labels(files: list[Path], chunksize: int = CHUNKSIZE) -> CorpusIndex:
    """Stream the ``Label`` column of every file and build the row index.

    Reading a single string column keeps this pass cheap: pandas still has
    to tokenise each line, but it materialises only one column per chunk.
    """
    logger.info("Pass 1/2: scanning labels in %d file(s)", len(files))

    # Vocabulary is discovered from the data rather than hard-coded, so a
    # re-release with a new attack type is picked up automatically.
    vocab: dict[str, int] = {}
    per_file_codes: list[np.ndarray] = []

    for path in files:
        parts: list[np.ndarray] = []
        # usecols by *name* works despite 02-20's wider schema, because the
        # column is called "Label" in every export.
        reader = pd.read_csv(
            path,
            usecols=[LABEL_COL],
            dtype={LABEL_COL: "string"},
            chunksize=chunksize,
            **_READ_OPTS,
        )
        for chunk in reader:
            parts.append(_encode_chunk(chunk[LABEL_COL], vocab))

        codes = np.concatenate(parts) if parts else np.empty(0, dtype=np.int16)
        per_file_codes.append(codes)
        logger.info("  %-18s %10s rows", path.name, f"{codes.size:,}")

    # Remap first-seen codes to alphabetical order so class indices are
    # stable no matter which order the files were read in.
    discovered = np.array(list(vocab), dtype=str)
    classes = np.sort(discovered)
    remap = np.empty(len(discovered) + 1, dtype=np.int16)
    remap[0] = -1  # index 0 holds the sentinel for invalid rows
    for name, old_code in vocab.items():
        remap[old_code + 1] = int(np.searchsorted(classes, name))

    file_indices = [
        FileIndex(path=path, n_rows=codes.size, codes=remap[codes + 1])
        for path, codes in zip(files, per_file_codes, strict=True)
    ]
    del per_file_codes

    offsets = np.concatenate([[0], np.cumsum([f.n_rows for f in file_indices])]).astype(np.int64)
    index = CorpusIndex(files=file_indices, classes=classes, offsets=offsets)

    n_invalid = sum(int((f.codes < 0).sum()) for f in file_indices)
    if n_invalid:
        logger.warning(
            "Marked %s unusable row(s) (embedded headers / empty labels); they keep their "
            "position so the two passes stay aligned, but are never sampled",
            f"{n_invalid:,}",
        )

    logger.info(
        "Pass 1 complete: %s rows, %d classes, %s usable",
        f"{index.n_rows:,}",
        len(classes),
        f"{index.n_rows - n_invalid:,}",
    )
    return index


# ----------------------------------------------------------------------
# Sampling plan
# ----------------------------------------------------------------------
def plan_stratified_sample(
    index: CorpusIndex,
    n_samples: int = SAMPLE_SIZE,
    min_per_class: int = MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Choose ``n_samples`` global row positions with stratified sampling.

    Uses ``StratifiedShuffleSplit`` -- **not** simple random sampling -- so
    the sample preserves the corpus class proportions exactly.

    A pure proportional draw gives ultra-rare classes too few rows to
    survive the later 70/30 stratified split (SQL Injection has ~87 rows in
    13M, i.e. ~2 rows at a 2.3% sampling rate, and scikit-learn refuses to
    stratify a class with fewer than 2 members). When ``min_per_class > 0``
    such classes are topped up afterwards, and the excess is taken back
    from the largest class so the total stays exactly ``n_samples``. The
    adjustment touches only a handful of rows, so proportions elsewhere are
    unaffected.
    """
    codes = index.codes
    # Restrict the sampling universe to usable rows.
    valid_pos = np.flatnonzero(codes >= 0)
    y = codes[valid_pos]

    if n_samples >= len(valid_pos):
        raise ValueError(
            f"Requested {n_samples:,} rows but only {len(valid_pos):,} usable rows exist"
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=random_state
    )
    # StratifiedShuffleSplit only inspects X's length; the positions
    # themselves ride along as a single column.
    sample_idx, _ = next(splitter.split(valid_pos.reshape(-1, 1), y))
    selected = np.sort(valid_pos[sample_idx])

    if min_per_class > 0:
        selected = _enforce_min_per_class(
            selected, codes, valid_pos, index.classes, min_per_class, random_state
        )

    _log_distribution(codes[selected], index.classes, total_codes=y)
    return selected


def plan_nested_sample(
    index: CorpusIndex,
    n_samples: int = SAMPLE_SIZE,
    min_per_class: int = MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Stratified sample whose sizes nest: 300k subset of 500k subset of 1M.

    Textbook stratified sampling done explicitly, in two steps:

    1. **Shuffle each stratum once.** The permutation of a class depends
       only on ``random_state``, never on ``n_samples``, so class *c* has
       one fixed ordering across every run.
    2. **Take a prefix.** Each class contributes ``quota[c]`` rows -- the
       *first* ``quota[c]`` of its ordering. Because the quotas grow
       monotonically with ``n_samples`` (see :func:`_proportional_quota`),
       a prefix for 300k is by construction a prefix of the one for 500k.

    That nesting is what makes a sample-size ladder interpretable: the
    only difference between two runs is how much data there is, not which
    rows were drawn. It also means the whole selection is reproducible
    from the seed alone -- rerunning picks the identical rows.
    """
    codes = index.codes
    valid_pos = np.flatnonzero(codes >= 0)
    y = codes[valid_pos]

    if n_samples > len(valid_pos):
        raise ValueError(
            f"Requested {n_samples:,} rows but only {len(valid_pos):,} usable rows exist"
        )

    class_counts = np.bincount(y, minlength=len(index.classes))
    quotas = _proportional_quota(class_counts, n_samples, min_per_class, index.classes)

    # One permutation per class, consumed in a fixed order so the random
    # stream is identical no matter what n_samples is.
    rng = np.random.default_rng(random_state)
    chosen = [rng.permutation(valid_pos[y == code])[: quotas[code]] for code in range(len(quotas))]

    selected = np.sort(np.concatenate(chosen))
    _log_distribution(codes[selected], index.classes, total_codes=y)
    return selected


def _proportional_quota(
    class_counts: np.ndarray,
    n_samples: int,
    min_per_class: int,
    classes: np.ndarray,
) -> np.ndarray:
    """Rows to draw from each class: proportional, floored, exactly ``n_samples``.

    Every minority class gets ``floor(n_samples * p_c)``, raised to
    ``min_per_class`` and capped at what the corpus holds. The largest
    class then absorbs whatever is left over, in either direction: flooring
    normally leaves a surplus, but raising ultra-rare classes to the floor
    can instead overshoot ``n_samples``, and the majority class gives those
    rows back.

    Balancing on the largest class -- rather than spreading the remainder by
    fractional part, as largest-remainder apportionment would -- is what
    makes the quotas grow monotonically with ``n_samples``, and monotone
    quotas are what make the samples nest. Largest-remainder is fairer for a
    single run but can *shrink* a class's quota as the total grows (the
    Alabama paradox), which would break nesting outright.

    The guarantee is exact for any two sizes more than a few dozen rows
    apart: the minority quotas together grow at rate ``1 - p_largest < 1``,
    so the majority quota strictly increases. Only adjacent values of
    ``n_samples`` could in principle tie, which no realistic ladder hits.
    """
    total = int(class_counts.sum())
    floors = np.minimum(min_per_class, class_counts)
    proportional = np.floor(n_samples * class_counts / total).astype(np.int64)
    quotas = np.clip(proportional, floors, class_counts)

    for code in np.flatnonzero(quotas > proportional):
        logger.warning(
            "Class %-24s would get %d row(s) proportionally; raised to the floor of %d",
            classes[code],
            int(proportional[code]),
            int(quotas[code]),
        )

    # The majority class takes the balance, so the total is exact.
    largest = int(np.argmax(class_counts))
    balance = n_samples - int(quotas.sum() - quotas[largest])

    if balance < floors[largest]:
        raise ValueError(
            f"min_per_class={min_per_class} reserves more than the {n_samples:,} rows "
            f"requested; lower --min-per-class or raise --sample-size"
        )
    if balance > class_counts[largest]:
        raise ValueError(
            f"Cannot reach {n_samples:,} rows: class {classes[largest]!r} would need "
            f"{balance:,} of its {class_counts[largest]:,} rows"
        )
    quotas[largest] = balance

    return quotas


def plan_sample(
    index: CorpusIndex,
    n_samples: int = SAMPLE_SIZE,
    mode: str = "nested",
    min_per_class: int = MIN_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Dispatch to the requested stratified sampling strategy."""
    if mode == "nested":
        return plan_nested_sample(index, n_samples, min_per_class, random_state)
    if mode == "independent":
        return plan_stratified_sample(index, n_samples, min_per_class, random_state)
    raise ValueError(f"Unknown sampling mode {mode!r}; use 'nested' or 'independent'")


def _enforce_min_per_class(
    selected: np.ndarray,
    codes: np.ndarray,
    valid_pos: np.ndarray,
    classes: np.ndarray,
    min_per_class: int,
    random_state: int,
) -> np.ndarray:
    """Top up under-represented classes, trimming the majority to compensate."""
    rng = np.random.default_rng(random_state)
    sel_set = set(selected.tolist())
    counts = np.bincount(codes[selected], minlength=len(classes))
    corpus_counts = np.bincount(codes[valid_pos], minlength=len(classes))

    additions: list[int] = []
    for code, count in enumerate(counts):
        # A class can never be topped up beyond what the corpus contains.
        deficit = min(min_per_class, corpus_counts[code]) - count
        if deficit <= 0:
            continue
        # Only scan the full code vector for the handful of starved classes.
        available = np.flatnonzero(codes == code)
        pool = np.array([p for p in available if p not in sel_set], dtype=np.int64)
        extra = rng.choice(pool, size=deficit, replace=False)
        additions.extend(extra.tolist())
        logger.warning(
            "Class %-24s under-sampled (%d rows); topped up to %d",
            classes[code],
            count,
            count + deficit,
        )

    if not additions:
        return selected

    # Remove the same number of rows from the most populous class so the
    # final sample size is still exactly what was requested.
    majority = int(np.argmax(counts))
    majority_rows = selected[codes[selected] == majority]
    drop = rng.choice(majority_rows, size=len(additions), replace=False)

    kept = np.setdiff1d(selected, drop, assume_unique=True)
    return np.sort(np.concatenate([kept, np.array(additions, dtype=np.int64)]))


def _log_distribution(
    sample_codes: np.ndarray, classes: np.ndarray, total_codes: np.ndarray
) -> None:
    """Print the sampled vs. original class distribution side by side."""
    sample_counts = np.bincount(sample_codes, minlength=len(classes))
    total_counts = np.bincount(total_codes, minlength=len(classes))
    logger.info("Stratified sample of %s rows:", f"{sample_codes.size:,}")
    logger.info("  %-26s %10s %8s %10s %8s", "class", "sampled", "%", "corpus", "%")
    for i, name in enumerate(classes):
        logger.info(
            "  %-26s %10s %7.3f%% %10s %7.3f%%",
            name,
            f"{sample_counts[i]:,}",
            100 * sample_counts[i] / sample_codes.size,
            f"{total_counts[i]:,}",
            100 * total_counts[i] / total_codes.size,
        )


# ----------------------------------------------------------------------
# Pass 2 -- materialise the chosen rows
# ----------------------------------------------------------------------
def materialize_sample(
    index: CorpusIndex,
    selected: np.ndarray,
    chunksize: int = CHUNKSIZE,
) -> pd.DataFrame:
    """Re-read the CSVs and return only the rows chosen by the plan.

    Memory stays bounded because at most one chunk is held at a time; the
    slices kept from each chunk are tiny (~2% of it).
    """
    logger.info("Pass 2/2: extracting %s selected rows", f"{selected.size:,}")

    # Map each global position to (file, position within that file).
    file_of = np.searchsorted(index.offsets, selected, side="right") - 1

    frames: list[pd.DataFrame] = []
    for fi, finfo in enumerate(index.files):
        # local positions are sorted because `selected` is sorted.
        local = selected[file_of == fi] - index.offsets[fi]
        if local.size == 0:
            continue

        taken: list[pd.DataFrame] = []
        collected = 0
        base = 0
        reader = pd.read_csv(finfo.path, chunksize=chunksize, low_memory=False, **_READ_OPTS)
        for chunk in reader:
            hi = base + len(chunk)
            take = local[(local >= base) & (local < hi)]
            if take.size:
                taken.append(chunk.iloc[take - base].copy())
                collected += take.size
            base = hi
            if collected == local.size:
                break  # every wanted row from this file is already in hand
        del reader

        if collected != local.size:
            raise RuntimeError(
                f"{finfo.path.name}: wanted {local.size} rows but pass 2 only reached "
                f"{collected} after {base} rows (pass 1 counted {finfo.n_rows}); "
                "the two passes are misaligned"
            )

        part = pd.concat(taken, ignore_index=True)
        _verify_alignment(part, finfo, expected_codes=finfo.codes[local], classes=index.classes)
        frames.append(part)
        logger.info("  %-18s %8d rows extracted", finfo.path.name, collected)

    # Normalise labels once, on the sample rather than the whole corpus.
    sample = pd.concat(frames, ignore_index=True)
    sample[LABEL_COL] = normalise_labels(sample[LABEL_COL]).astype(str)
    sample = _normalise_dtypes(sample)
    logger.info("Pass 2 complete: %s x %d columns", f"{len(sample):,}", sample.shape[1])
    return sample


def _normalise_dtypes(sample: pd.DataFrame) -> pd.DataFrame:
    """Give every column one dtype across the whole sample.

    pandas infers dtypes per chunk, and a chunk that happens to contain one
    of the embedded header lines has its numeric columns inferred as
    ``object`` rather than ``int64``. Concatenating those chunks leaves
    columns holding both ``80`` and ``'80'``, which Parquet rejects
    outright and which would silently poison any downstream arithmetic.

    Text columns are pinned to string; everything else is coerced to a
    float32 numeric, matching what the preprocessor does later anyway and
    halving both the cache size and the peak RAM of a large sample.
    """
    text_cols = [c for c in (LABEL_COL, TIMESTAMP_COL, *TEXT_IDENTITY_COLS) if c in sample.columns]
    numeric_cols = [c for c in sample.columns if c not in text_cols]

    for col in text_cols:
        sample[col] = sample[col].astype("string")

    coerced = sample[numeric_cols].apply(pd.to_numeric, errors="coerce")
    n_bad = int(coerced.isna().sum().sum() - sample[numeric_cols].isna().sum().sum())
    if n_bad > 0:
        logger.warning(
            "Coerced %s non-numeric cell(s) to NaN across %d feature column(s)",
            f"{n_bad:,}",
            len(numeric_cols),
        )
    sample[numeric_cols] = coerced.astype(np.float32)
    return sample


def _verify_alignment(
    part: pd.DataFrame,
    finfo: FileIndex,
    expected_codes: np.ndarray,
    classes: np.ndarray,
) -> None:
    """Fail loudly if pass 2 read different rows than pass 1 planned.

    Row positions are the one thing this design cannot get wrong silently,
    so the labels of the extracted rows are compared against the labels
    recorded during pass 1 at those exact positions.
    """
    got = normalise_labels(part[LABEL_COL]).to_numpy(dtype=object)
    want = classes[expected_codes]
    mismatch = int((got != want).sum())
    if mismatch:
        raise RuntimeError(
            f"{finfo.path.name}: {mismatch} of {len(part)} extracted rows have a label that "
            "differs from pass 1 -- chunk boundaries drifted between passes"
        )
