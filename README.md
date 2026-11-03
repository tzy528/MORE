# MORE: Multi-Order Response Energy for Graph Anomaly Detection

MORE (Multi-Order Response Energy) uses node-wise deviations from typical response energy levels across multiple graph responses as label-free anomaly evidence. It incorporates graph relationships through triplet-guided learning to obtain more discriminative anomaly scores.

This repository runs **one dataset at a time**. By default, it runs `yelpchi` for **30 epochs** with **early stopping disabled**. Nodes are stratified into training and test sets at an **80:20 ratio**, without a validation set. The model from the final epoch is used for testing.

## 1. Requirements

- Python 3.10 or later.
- PyTorch 2.1 or later, with CPU or CUDA support.
- NumPy, SciPy, and scikit-learn; see `requirements.txt`.
- DGL compatible with your PyTorch/CUDA installation, if loading DGL graph binaries.
- `h5py`, if loading MATLAB v7.3 files.

First install [PyTorch](https://pytorch.org/get-started/locally/) for your hardware, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For DGL data files, follow the [DGL installation guide](https://www.dgl.ai/pages/start.html) to select a compatible version. DGL is not required for standard MAT/NPZ files. MATLAB v7.3 support can be installed separately:

```bash
python -m pip install h5py
```

## 2. Dataset Download and Preparation

Dataset files are not included in this repository. Only the dataset selected for the current run needs to be available locally.

### Dataset sources

The datasets other than **Flickr, Weibo, and Questions** can be downloaded from the GitHub repository accompanying **A Label-Free Heterophily-Guided Approach for Unsupervised Graph Fraud Detection**:

- [HUGE-GAD](https://github.com/CampanulaBells/HUGE-GAD)

For the five datasets used in the MORE paper, use the following sources and check that the downloaded versions match these statistics:

| Dataset | Nodes | Anomalous nodes | Attributes | Source |
| --- | ---: | ---: | ---: | --- |
| Flickr | 7,575 | 450 | 12,047 | [Deep Anomaly Detection on Attributed Networks](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch/tree/main) |
| YelpChi | 23,831 | 1,217 | 32 | [HUGE-GAD](https://github.com/CampanulaBells/HUGE-GAD) |
| Weibo | 8,405 | 868 | 400 | [GAL](https://github.com/DM2-ND/GAL); convert the files as described below |
| Questions | 48,921 | 1,460 | 301 | [heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs) |
| YelpChiFull | 45,954 | 6,677 | 32 | [HUGE-GAD](https://github.com/CampanulaBells/HUGE-GAD) |

**YelpChi and YelpChiFull are different dataset versions.** The default `yelpchi` run expects the version with 23,831 nodes. Use `--dataset YelpChiFull` for the version with 45,954 nodes. Renaming a file does not make the two versions equivalent.

### Weibo

The Weibo dataset can be downloaded from the repository accompanying **Error-Bounded Graph Anomaly Loss for GNNs**:

- [GAL](https://github.com/DM2-ND/GAL)

Use the preprocessing script `convert_weibo_to_mat(1).py` in the `datasets/` directory to combine the following five files into a single `.mat` file:

- `weibo_s_features_bow.pkl`
- `weibo_s_features_loc.pkl`
- `weibo_s_graph_u2p.pkl`
- `weibo_s_graph_u2u.pkl`
- `weibo_s_labels_u.pkl`

Ensure that the conversion script is present in `datasets/`, configure its input and output paths for your local files, and save the converted graph as `datasets/Weibo.mat`. The training script reads the converted graph rather than the five raw pickle files.

### Questions

The Questions dataset can be downloaded from the repository accompanying **A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?**:

- [heterophilous-graphs](https://github.com/yandex-research/heterophilous-graphs)

The upstream datasets are provided in its `data/` directory. Place the Questions NPZ file in this repository's `datasets/` directory, for example as `datasets/Questions.npz`.

The upstream file stores each undirected edge once. The MORE loader symmetrizes the adjacency matrix automatically. It also creates a new 80:20 stratified split instead of using the upstream split masks.

### Local files and supported formats

Place your data in `datasets/`, for example `datasets/yelpchi.mat`, `datasets/Weibo.mat`, `datasets/Questions.npz`, or `datasets/YelpChiFull`. Alternatively, specify another directory with `--data-dir`.

The loader detects DGL binaries by their file header, including files with a `.mat` extension. It prints the actual node count, anomaly count, and attribute dimension after preprocessing at startup.

Supported formats:

- `.mat`: feature keys include `Attributes`, `features`, and `X`; adjacency keys include `Network`, `homo`, and `adj`; label keys include `Label`, `labels`, and `y`.
- `.npz` / `.npy`: graph dictionaries containing attributes, an adjacency matrix or `edge_index`, and labels. Load only trusted files.
- DGL graph binaries: `.bin`, extensionless files, or files identified by the DGL header. Node feature keys include `feat`, `feature`, `features`, and `x`; label keys include `label`, `labels`, and `y`. Use homogeneous graph versions.

Labels are encoded as `0` for genuine nodes, `1` for anomalous nodes, and `-1` for unknown labels. Ground-truth labels are used only for stratified splitting and final evaluation, not for training losses, reference statistics, or triplet sampling. Nodes with unknown labels are added to the unsupervised training set and excluded from evaluation. Thus, the **80:20 ratio applies to nodes with known labels**. When all labels are known, it applies to all nodes, subject to scikit-learn's integer rounding.

## 3. Running the Code

Run commands from the repository root. The default command is:

```bash
python run_fast.py
```

The equivalent explicit command is:

```bash
python run_fast.py --data-dir datasets --dataset yelpchi --epochs 30 --seeds 42 --output-dir runs/yelpchi
```

To run YelpChiFull downloaded from HUGE-GAD:

```bash
python run_fast.py --data-dir datasets --dataset YelpChiFull --epochs 30 --seeds 42 --output-dir runs/yelpchifull
```

To switch datasets, specify only one dataset in each command:

```bash
python run_fast.py --dataset Weibo --epochs 30 --output-dir runs/weibo
python run_fast.py --dataset Flickr --epochs 10 --output-dir runs/flickr
python run_fast.py --dataset Questions --epochs 10 --output-dir runs/questions
```

Multiple random seeds can be used for the same dataset:

```bash
python run_fast.py --dataset yelpchi --epochs 30 --seeds 42 43 44 45 46 --output-dir runs/yelpchi_5seeds
```

The five seeds above are examples, not a verified list of the seeds used in the original experiments. The summary file reports the mean and standard deviation across the specified seeds.

The device is selected automatically: CUDA when available, otherwise CPU. You can also select it explicitly:

```bash
python run_fast.py --dataset yelpchi --device cuda:0 --output-dir runs/yelpchi_cuda
python run_fast.py --dataset yelpchi --device cpu --output-dir runs/yelpchi_cpu
```

Use a new output directory for each run; existing experiment directories are not overwritten. `--datasets` is retained as an alias for `--dataset`, but it also accepts only one dataset. Neither `all` nor multiple dataset names are accepted. Run `python run_fast.py --help` to see all options.

## 4. Default Settings and Implementation

| Setting | Default |
| --- | --- |
| Dataset | yelpchi |
| Training epochs | 30 |
| Early stopping | Disabled; no validation set or best-epoch selection |
| Split | Stratified training:test = 80:20 |
| Final model | Model from the final epoch |
| Model variant | `homo`, the complete MORE model |
| Graph computation | `full`, exact full-graph computation |
| Hidden / representation dimension | 64 / 32 |
| Maximum order K / score cap c | 3 / 20 |
| Learning rate / weight decay | 0.001 / 0.0005 |
| q0 / δsame / δrank | 0.5 / 0.1 / 0.25 |
| Ranking margin / relation margin / temperature | 0.1 / 0.1 / 0.1 |
| λrank / λhom / λvar | 1 / 1 / 0.05 |
| Candidate-pool refresh / score-statistics logging | Every 5 epochs / every 5 epochs |
| Parameter updates per epoch | 1 by default in full-graph mode |

The main implementation follows Eqs. (4)-(17) in the paper: MLP encoding; Chebyshev responses using `L-I = -D^(-1/2) A D^(-1/2)`; scoring with orders 1 through K; `log1p` of squared response norms; training-node median/MAD normalization; the maximum absolute deviation across orders with score clipping; triplets consisting of a connected low-deviation pair and a higher-deviation non-neighbor; and the ranking, triplet relation, and representation variance losses.

Reference statistics and candidate pools are refreshed periodically. Statistics remain fixed within each refresh interval, and gradients are not propagated through median/MAD estimation or sampling. Training logs contain losses and score distributions. Test AUROC and AUPRC (average precision) are computed once after training and are not used for model selection. Add `--no-evaluate-test` to save the model and scores without computing test metrics.

By default, attribute preprocessing removes constant columns and applies StandardScaler. Sparse attributes are scaled without centering; dense attributes are centered. This step uses all observed node attributes without labels. Removing constant columns may reduce the input dimension relative to the raw dataset statistics in the paper. Use `--preprocessing legacy` to retain constant columns for comparison.

If GPU memory is insufficient, bounded subgraph computation can be enabled explicitly:

```bash
python run_fast.py --dataset YelpChiFull --graph-mode bounded --gpu-budget-gb 8 --cache-gb 2 --core-nodes 2048 --halo-nodes 16384 --halo-edges 2000000 --output-dir runs/yelpchifull_bounded
```

**Bounded mode truncates neighborhoods/edges and restricts triplet candidates to each core, so it is not strictly equivalent to the full-graph formulation in the paper.** `--graph-mode auto` may also select bounded mode. Use the default `full` mode when comparing against the paper. `--gpu-budget-gb` controls a planning estimate, not a hard GPU memory limit.

## 5. Output Files

| File | Contents |
| --- | --- |
| `config.json` | Complete run configuration |
| `results.csv` / `summary.csv` | Per-seed results and aggregate statistics |
| `execution_plan.json` | Full-graph/subgraph execution mode and memory estimates |
| `feature_audit.json` | Attribute statistics after preprocessing |
| `split_counts.json` | Node and anomaly counts for train/val/test; val contains 0 nodes |
| `history.csv` | Per-epoch losses, constraint counts, and parameter updates |
| `score_history.csv` | Score distributions, reference statistics, and clipping fractions; no test metrics |
| `final_scores.npz` | Final scores, signed deviations by order, reference statistics, and split indices; val is empty |
| `model.pt` | Final-epoch model parameters and run configuration |
| `result.json` | Individual run results, including the final epoch and early_stop=false |

## 6. Checks

```bash
python -m unittest -v test_fast.py
```

The CPU checks cover attribute preprocessing, the 80:20 split, triplet sampling and weights, response energies, and normalization. With PyTorch installed, additional checks cover numerical agreement, gradients, loss formulas, the single-dataset interface, and end-to-end execution on small graphs. Tests requiring PyTorch are skipped when it is unavailable.

The environment used to prepare this package did not include PyTorch/DGL or the real datasets. Real-dataset training, GPU memory usage, and the numerical results reported in the paper have therefore not been validated in that environment. See `PAPER_ALIGNMENT.md` for the implementation review.
