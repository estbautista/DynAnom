# Modifications and provenance

This repository adapts the graph-level code from [DynAnom](https://github.com/zjlxgxz/DynAnom), by **Xingzhi Guo, Baojian Zhou, and Steven Skiena** (*Subset Node Anomaly Tracking over Large Dynamic Graphs*, KDD 2022, DOI [10.1145/3534678.3539389](https://doi.org/10.1145/3534678.3539389)). The adaptation was used in the MuTGN experimental evaluation.

The changes below describe the differences between the upstream DynAnom implementation and the files retained in this repository. The cleaned implementation was rebuilt, and both the PPR sanity check and the Wikipedia experiment were successfully rerun on 18 November 2026.

## Main graph-level experiment

**File:** `src/algorithms/DynAnom/exp_DynAnom_graph_level.py`

- Added YAML-based configuration for input/output paths, PPR parameter `alpha`, window size/shift, and train/test ratio.
- Added a weighted temporal stream reader using the separately authored `lsmanip` package. Window assignments are computed directly in the experiment script: interactions are assigned to each matching window, and weights are converted to integers and expanded into repeated events in a temporary comma-separated file.
- Retained **only** DynAnom's existing `DynAnomScore(l2-sum)` graph-level score. Other upstream score variants and the associated hashed embedding computations were disabled.
- Disabled upstream CSV and top-k ranking evaluation and instead pickle-serialized the resulting numerical scores for external evaluation.
- Updated snapshot-edge iteration to check the end of the input, process the current edge before incrementing the cursor, and include events at the current snapshot boundary. This changes the upstream boundary convention and aligns it with MuTGN splitting.
- Added handling for empty shared-node sets and empty sparse PPR representations.
- Adjusted progress display and integration details for the experiments.

## Event reader and snapshot construction

**File:** `src/graph_data_utils.py`

- Adapted `read_data()` to accept the temporary headerless CSV `source,destination,window_index`, rather than the upstream space-separated `timestamp source destination anomaly_label` format.
- Sets internal event labels to zero; ground-truth anomaly labels are not supplied to DynAnom and are evaluated externally.
- Preserves window indices instead of subtracting the first event timestamp.
- Enumerates successive window indices, including windows with no new interactions, rather than only effective snapshots with edge changes.
- Returns six values instead of the original seven, omitting the upstream anomalous-node description set.

Other upstream helper functions remain in this file, although the graph-level baseline does not use the anomaly-generation helpers.

## Core incremental PPR algorithm

**File:** `src/algorithms/DynAnom/DynamicPPE.py`

The per-edge `tqdm` progress bar was disabled.

## Added code and research utilities

- `src/algorithms/DynAnom/config.py`: YAML configuration loading and output-directory creation.
- `src/algorithms/DynAnom/lsmanip/`: separately authored temporal link-stream utilities; its source code is made available under EUPL-1.2.
- `src/algorithms/DynAnom/scores_train_data.py`: utility for inspecting anomaly scores over the training/validation split.
- `Dockerfile` and `requirements-reconstructed.txt`: containerized Python 3.7 execution environment with pinned dependencies reconstructed for reproducibility.

## Scope of this distribution

The focused DynAnomAgg distribution omits the upstream node-level experiment, case study, synthetic-anomaly generation utility, visualization notebook, and legacy experiment scripts. Obsolete exploratory launchers and intermediate backups were also removed.

The original DynAnom README is retained verbatim as `UPSTREAM_README.md`. The original GPLv3 `LICENSE` is retained at the repository root. The `lsmanip` EUPL 1.2 license text is provided separately in `src/algorithms/DynAnom/lsmanip/LICENSE.txt`.
