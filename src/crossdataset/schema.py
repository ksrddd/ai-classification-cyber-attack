"""Shared feature schema for CICIDS2017 <-> CSE-CIC-IDS2018 work.

The two datasets ship the same CICFlowMeter measurements under different
column names -- ``Total Fwd Packets`` in 2017 is ``Tot Fwd Pkts`` in 2018 --
so nothing can be trained on one and tested on the other until the names are
reconciled. This module is that reconciliation.

Canonical names are the 2017 spellings, because 2017 is the project's primary
dataset and its names already appear in ``src.data.schema.EXPECTED_FEATURES``,
in saved model artefacts, and in the inference contract.

How the pairing was derived
---------------------------
Not by matching names, which would have mis-paired the twelve columns that
reorder their words (``Min Packet Length`` vs ``Pkt Len Min``). Both files
come out of CICFlowMeter and preserve its emission order, so the columns were
listed in file order and paired by position, then read back to confirm every
pair was semantically the same measurement. The orders agree position for
position with no crossings.

Three groups are excluded before pairing, leaving 77 on each side -- the same
count the advisor's protocol specifies:

* 2017 ``Fwd Header Length.1`` -- a duplicate of ``Fwd Header Length`` and a
  known defect in the published CSVs.
* 2018 ``Flow ID`` / ``Src IP`` / ``Src Port`` / ``Dst IP`` -- identifiers. A
  model that learns "traffic from 18.219.211.138 is an attack" has memorised
  the lab.
* 2018 ``Protocol`` and ``Timestamp`` -- absent from the 2017 feature list.
  ``Timestamp`` is also target leakage, since each attack ran in its own
  capture window. Dropping ``Protocol`` was measured (2026-08-09) to cost
  nothing: three of four models moved less than 0.002 macro-F1.

The ``Destination Port`` disagreement
-------------------------------------
``CANONICAL_FEATURES`` includes ``Destination Port`` because the 2017 pipeline
trains on it. The 2018 pipeline drops it by default as lab-tied -- HOIC always
targeted port 80. Both positions are defensible and the two pipelines
currently disagree, so a cross-dataset run must choose one and apply it to
both sides. Measured on the 300k sample, keeping it lifts Infilteration F1
from 0.1185 to 0.1703; it is also the most likely single dataset fingerprint
and so the first candidate for adversarial validation. Use
:func:`canonical_features` to get the list with or without it.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: 2017 columns excluded before pairing.
DROPPED_IDS2017: tuple[str, ...] = ("Fwd Header Length.1",)

#: 2018 columns excluded before pairing.
DROPPED_IDS2018: tuple[str, ...] = (
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Protocol",
    "Timestamp",
)

#: The canonical name of the destination-port column, which the two pipelines
#: currently treat differently. See the module docstring.
DST_PORT = "Destination Port"

CANONICAL_FEATURES: tuple[str, ...] = (
    'Destination Port',
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min',
)

IDS2018_TO_CANONICAL: dict[str, str] = {
    'Dst Port'         : 'Destination Port',
    'Flow Duration'    : 'Flow Duration',
    'Tot Fwd Pkts'     : 'Total Fwd Packets',
    'Tot Bwd Pkts'     : 'Total Backward Packets',
    'TotLen Fwd Pkts'  : 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts'  : 'Total Length of Bwd Packets',
    'Fwd Pkt Len Max'  : 'Fwd Packet Length Max',
    'Fwd Pkt Len Min'  : 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean' : 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std'  : 'Fwd Packet Length Std',
    'Bwd Pkt Len Max'  : 'Bwd Packet Length Max',
    'Bwd Pkt Len Min'  : 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean' : 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std'  : 'Bwd Packet Length Std',
    'Flow Byts/s'      : 'Flow Bytes/s',
    'Flow Pkts/s'      : 'Flow Packets/s',
    'Flow IAT Mean'    : 'Flow IAT Mean',
    'Flow IAT Std'     : 'Flow IAT Std',
    'Flow IAT Max'     : 'Flow IAT Max',
    'Flow IAT Min'     : 'Flow IAT Min',
    'Fwd IAT Tot'      : 'Fwd IAT Total',
    'Fwd IAT Mean'     : 'Fwd IAT Mean',
    'Fwd IAT Std'      : 'Fwd IAT Std',
    'Fwd IAT Max'      : 'Fwd IAT Max',
    'Fwd IAT Min'      : 'Fwd IAT Min',
    'Bwd IAT Tot'      : 'Bwd IAT Total',
    'Bwd IAT Mean'     : 'Bwd IAT Mean',
    'Bwd IAT Std'      : 'Bwd IAT Std',
    'Bwd IAT Max'      : 'Bwd IAT Max',
    'Bwd IAT Min'      : 'Bwd IAT Min',
    'Fwd PSH Flags'    : 'Fwd PSH Flags',
    'Bwd PSH Flags'    : 'Bwd PSH Flags',
    'Fwd URG Flags'    : 'Fwd URG Flags',
    'Bwd URG Flags'    : 'Bwd URG Flags',
    'Fwd Header Len'   : 'Fwd Header Length',
    'Bwd Header Len'   : 'Bwd Header Length',
    'Fwd Pkts/s'       : 'Fwd Packets/s',
    'Bwd Pkts/s'       : 'Bwd Packets/s',
    'Pkt Len Min'      : 'Min Packet Length',
    'Pkt Len Max'      : 'Max Packet Length',
    'Pkt Len Mean'     : 'Packet Length Mean',
    'Pkt Len Std'      : 'Packet Length Std',
    'Pkt Len Var'      : 'Packet Length Variance',
    'FIN Flag Cnt'     : 'FIN Flag Count',
    'SYN Flag Cnt'     : 'SYN Flag Count',
    'RST Flag Cnt'     : 'RST Flag Count',
    'PSH Flag Cnt'     : 'PSH Flag Count',
    'ACK Flag Cnt'     : 'ACK Flag Count',
    'URG Flag Cnt'     : 'URG Flag Count',
    'CWE Flag Count'   : 'CWE Flag Count',
    'ECE Flag Cnt'     : 'ECE Flag Count',
    'Down/Up Ratio'    : 'Down/Up Ratio',
    'Pkt Size Avg'     : 'Average Packet Size',
    'Fwd Seg Size Avg' : 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg' : 'Avg Bwd Segment Size',
    'Fwd Byts/b Avg'   : 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg'   : 'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg' : 'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg'   : 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg'   : 'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg' : 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts' : 'Subflow Fwd Packets',
    'Subflow Fwd Byts' : 'Subflow Fwd Bytes',
    'Subflow Bwd Pkts' : 'Subflow Bwd Packets',
    'Subflow Bwd Byts' : 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min' : 'min_seg_size_forward',
    'Active Mean'      : 'Active Mean',
    'Active Std'       : 'Active Std',
    'Active Max'       : 'Active Max',
    'Active Min'       : 'Active Min',
    'Idle Mean'        : 'Idle Mean',
    'Idle Std'         : 'Idle Std',
    'Idle Max'         : 'Idle Max',
    'Idle Min'         : 'Idle Min',
}

#: 2017 needs no renaming -- its names are the canonical ones.
IDS2017_TO_CANONICAL: dict[str, str] = {name: name for name in CANONICAL_FEATURES}

_RENAMERS = {"ids2017": IDS2017_TO_CANONICAL, "ids2018": IDS2018_TO_CANONICAL}


def canonical_features(*, keep_dst_port: bool = True) -> tuple[str, ...]:
    """Return the shared feature names, optionally without the port column.

    ``keep_dst_port=False`` yields the 76 columns both pipelines agree on.
    """
    if keep_dst_port:
        return CANONICAL_FEATURES
    return tuple(name for name in CANONICAL_FEATURES if name != DST_PORT)


def to_canonical(
    df: pd.DataFrame, dataset: str, *, keep_dst_port: bool = True
) -> pd.DataFrame:
    """Rename ``df`` into the shared schema and return the features in order.

    Parameters
    ----------
    df:
        A frame from either dataset. Extra columns -- labels, timestamps,
        identifiers -- are ignored rather than rejected, so a caller may hand
        over a whole loaded file.
    dataset:
        ``"ids2017"`` or ``"ids2018"``.
    keep_dst_port:
        Keep ``Destination Port``. Both sides of a cross-dataset run must pass
        the same value or they are not comparable.

    Raises
    ------
    KeyError
        If any expected column is missing. A silent reindex would fill the gap
        with NaN, the imputer would replace it with a median, and the run would
        finish and report a plausible score for a feature that was never there.
    """
    try:
        renamer = _RENAMERS[dataset]
    except KeyError:
        raise ValueError(
            f"dataset must be 'ids2017' or 'ids2018'; got {dataset!r}"
        ) from None

    wanted = canonical_features(keep_dst_port=keep_dst_port)
    renamed = df.rename(columns=renamer)
    missing = [name for name in wanted if name not in renamed.columns]
    if missing:
        raise KeyError(
            f"{dataset}: {len(missing)} expected column(s) missing after rename: "
            f"{missing}"
        )
    logger.info(
        "%s -> canonical schema: %d feature(s), Destination Port %s",
        dataset,
        len(wanted),
        "kept" if keep_dst_port else "dropped",
    )
    return renamed.loc[:, list(wanted)]
