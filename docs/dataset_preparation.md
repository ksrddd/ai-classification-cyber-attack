# Dataset Preparation

## Source

The Canadian Institute for Cybersecurity's CICIDS2017 and CSE-CIC-IDS2018 datasets.
Dataset pages:

- <https://www.unb.ca/cic/datasets/ids-2017.html>
- <https://www.unb.ca/cic/datasets/ids-2018.html>

UNB CIC ships the data as PCAPs (raw packet captures) and as
pre-extracted flow CSVs. **We consume the CSVs**; PCAP parsing is
explicitly out of scope.

## Files

Inside `MachineLearningCSV.zip` -> folder `MachineLearningCVE/`:

| File | Day | Attacks present |
|------|-----|-----------------|
| `Monday-WorkingHours.pcap_ISCX.csv` | Mon | None (all BENIGN) |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tue | FTP-Patator, SSH-Patator |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wed | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thu AM | Web Attack (Brute Force, XSS, Sql Injection) |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thu PM | Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Fri AM | Bot |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Fri PM | PortScan |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Fri PM | DDoS |

Total: ~2.8M flow records, 78 features, 15 distinct raw labels.

## How to extract

```powershell
# Assuming the zip is at D:\CI-CIDS-2017\md5\CSVs\MachineLearningCSV.zip
Expand-Archive `
    -Path 'D:\CI-CIDS-2017\md5\CSVs\MachineLearningCSV.zip' `
    -DestinationPath 'data\raw\.cicids_tmp'
Move-Item data\raw\.cicids_tmp\MachineLearningCVE\*.csv data\raw\
Remove-Item -Recurse data\raw\.cicids_tmp
Get-ChildItem data\raw\*.csv
```

Or via Git Bash:

```bash
cd data/raw
unzip /d/CI-CIDS-2017/md5/CSVs/MachineLearningCSV.zip
mv MachineLearningCVE/*.csv .
rmdir MachineLearningCVE
ls -la *.csv
```

The project's loader picks up every `*.csv` under `data/raw/`
automatically; there's no explicit list to maintain. The total disk
footprint of the extracted CSVs is ~884 MB.

## Known data quirks (handled by the pipeline)

1. **Leading-space column names.** Every column ships as
   `" Destination Port"` etc. `clean_column_names` strips it.
2. **Duplicate column `Fwd Header Length.1`** -- a literal duplicate of
   `Fwd Header Length`. Cleaning drops it.
3. **`+/-Inf` in flow-rate columns** -- particularly `Flow Bytes/s` and
   `Flow Packets/s` when `Flow Duration` is zero. Cleaning replaces
   these with NaN, then drops the affected rows.
4. **NaN in flow-rate columns** -- same root cause as #3. Dropped.
5. **Exact duplicate rows** -- dropped.
6. **Web Attack labels embed Windows-1252 byte `0x96`** (en-dash) --
   decoded as `` when the CSV is read with `latin-1`. The label
   normalizer strips C0/C1 control chars before lookup.
7. **Inconsistent capitalisation** in `DoS Hulk` vs `DoS slowloris`.
   Label normalization lowercases everything.
8. **Misspelling**: `Infilteration` in the file name, `Infiltration` in
   the label column. The label map handles both.

## Label mapping

See [`feature_mapping.md`](feature_mapping.md) for the canonical
raw -> normalized label table.

## Memory

Loading every CSV uncompressed materializes ~1.5 GB of pandas data.
For development, set `data.subsample_n: 300000` in `config.yaml` --
the loader does a stratified subsample so even rare classes
(Heartbleed = 11 rows in total) survive.
