# DynAnom

This repository contains an adaptation of [DynAnom](https://github.com/zjlxgxz/DynAnom), developed by Xingzhi Guo, Baojian Zhou, and Steven Skiena (KDD 2022). It applies DynAnom's graph-level anomaly detection method to **graph snapshots formed by aggregating temporal interactions within consecutive time windows**.

It addresses the following research question: **Can window-level anomalies still be detected after collapsing the interactions within each window into a single graph?** This adaptation was used as a baseline in the experimental evaluation of **MuTGN**.

The adaptation supports configurable window sizes and shifts, preserves DynAnom's incremental personalized PageRank (PPR) algorithm and `DynAnomScore(l2-sum)` scoring function, and exports anomaly scores for external evaluation. It also includes safeguards for empty graph representations and edge-stream boundaries.

For details on the adaptations, see [MODIFICATIONS.md](MODIFICATIONS.md). The original project documentation is preserved in [UPSTREAM_README.md](UPSTREAM_README.md).

## Method

The adapted implementation reads a weighted temporal edge stream and groups interactions into windows of configurable size and shift. Each interaction contributes its weight to every window containing its timestamp, allowing windows to overlap.

The resulting window-indexed edge batches are processed sequentially by DynAnom's **incremental graph estimator**. The estimator maintains its graph state across windows rather than restarting independently for each window.

The baseline uses DynAnom's original **`DynAnomScore(l2-sum)`**: the sum of L2 distances between approximate PPR vectors for nodes shared by consecutive snapshots. Other scoring alternatives and the original top-k ranking evaluation are disabled. Anomaly scores are exported for external evaluation.

## Input and output

The input is a **headerless, comma-separated** weighted temporal edge stream with four columns:

```text
source,destination,timestamp,weight
```

For example:

```text
0,1,0,2
1,2,60,1
0,2,120,3
```

Source and destination identifiers and timestamps must be integers. Weights represent **integer interaction counts**: the current adapter converts each weight using `int(weight)` and expands it into repeated edges.

Timestamps must be non-negative, and `window_size` and `window_shift` must use the same time unit as the timestamps. The input does not contain anomaly labels.

The main experiment writes a Python pickle containing a NumPy array of window-level anomaly scores to the configured `paths.score_file`. Ground-truth labels and ROC AUC / PR AUC evaluation are handled outside this repository.

## Reproducible environment

The tested container uses **Python 3.7** with the pinned packages in [`requirements-reconstructed.txt`](requirements-reconstructed.txt). [Podman](https://podman.io/) is used below; the image can also be built with Docker.

From the root of this repository, build the image:

```bash
podman build --no-cache -t dynanomimg .
```

Run the original DynAnom PPR sanity check:

```bash
podman run --rm dynanomimg \
  python /app/src/algorithms/DynAnom/DynamicPPE.py
```

The approximate and exact PPR vectors should agree to the displayed precision. The example in the original DynAnom documentation is:

```text
appr. ppr  [0.474 0.081 0.222 0.143 0.081]
exact ppr  [0.474 0.081 0.222 0.143 0.081]
```

## Running window-level anomaly detection

The main entry point is:

```text
src/algorithms/DynAnom/exp_DynAnom_graph_level.py
```

It accepts a YAML configuration file through `--config`. For example, save the following as `config.yaml`:

```yaml
model:
  alpha: 0.15

splitting:
  train_test_ratio: 0.7
  window_size: 600
  window_shift: 600

paths:
  dataset: /data/your_dataset.csv
  score_file: /outputs/scores.pkl
```

Adjust the parameters and dataset filename as needed. Paths in the configuration refer to locations **inside the container**.

To run the detector, place the input CSV file in a local `data/` directory and create an `outputs/` directory. Then, from the repository root:

```bash
mkdir -p data outputs

podman run --rm \
  -v "$PWD/config.yaml:/config.yaml:ro,z" \
  -v "$PWD/data:/data:ro,z" \
  -v "$PWD/outputs:/outputs:Z" \
  dynanomimg \
  python /app/src/algorithms/DynAnom/exp_DynAnom_graph_level.py \
  --config /config.yaml
```

The detector reads the input edge stream, processes the aggregated windows sequentially, and writes the resulting anomaly scores to `outputs/scores.pkl`.

Temporary edge files are created inside the container under the source directory. The container's `/app/src` directory must therefore remain writable. Generated temporary files and score pickles are runtime artifacts, not source files.

Ground-truth labels and evaluation metrics are handled externally. The MuTGN repository documents the configurations and evaluation procedures used for the experiments reported in the paper.

## Repository contents

| Path | Purpose |
| --- | --- |
| `src/algorithms/DynAnom/exp_DynAnom_graph_level.py` | Main graph-level baseline and score export |
| `src/algorithms/DynAnom/DynamicPPE.py` | Upstream incremental PPR implementation; per-edge progress display disabled |
| `src/graph_data_utils.py` | Adapted event reader and snapshot construction |
| `src/algorithms/DynAnom/config.py` | YAML configuration helper |
| `src/algorithms/DynAnom/lsmanip/` | Link stream manipulation library, separately authored |
| `src/algorithms/DynAnom/scores_train_data.py` | Exploratory training/validation score analysis |

The original DynAnom node-level, case-study, synthetic-anomaly, visualization, and legacy launcher components are omitted from this focused distribution. They remain available in the upstream repository.

## Provenance and citation

If you use DynAnom, please cite the original DynAnom work:

```bibtex
@inproceedings{guo2022dynanom,
  title     = {Subset Node Anomaly Tracking over Large Dynamic Graphs},
  author    = {Guo, Xingzhi and Zhou, Baojian and Skiena, Steven},
  booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  year      = {2022},
  doi       = {10.1145/3534678.3539389}
}
```

For original experiments, background references, and upstream setup instructions, see [`UPSTREAM_README.md`](UPSTREAM_README.md) and the [DynAnom source repository](https://github.com/zjlxgxz/DynAnom).

## Licensing

- **DynAnom-derived implementation and combined DynAnom distribution:** GNU General Public License, version 3 (GPLv3). The original upstream license text is preserved in the root [`LICENSE`](LICENSE).
- **Separately authored `lsmanip` component:** European Union Public Licence, version 1.2 (EUPL-1.2). Its license text is in [`LICENSES/EUPL-1.2.txt`](LICENSES/EUPL-1.2.txt). The EUPL 1.2 compatibility provision permits inclusion in a combined GPLv3 work.

Existing original copyright and attribution notices remain applicable. Changes to upstream files are described in [`MODIFICATIONS.md`](MODIFICATIONS.md). These licenses do not imply endorsement by the original DynAnom authors.
