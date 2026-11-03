#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Weibo_s / Tencent-Weibo pickle files to a common .mat format for GAD baselines.

Expected input files under --input_dir:
  - weibo_s_graph_u2u.pkl        (preferred user-user graph; optional if graph_u2p exists)
  - weibo_s_graph_u2p.pkl        (user-post/hashtag graph; used to generate user-user graph if needed)
  - weibo_s_features_bow.pkl     (BOW/content features)
  - weibo_s_features_loc.pkl     (location/profile features; optional if --feature bow)
  - weibo_s_labels_u.pkl         (user labels)

Output .mat contains multiple common key aliases:
  - Network / A / net / homo: sparse user-user adjacency
  - Attributes / X / features: user features
  - Label / gnd / labels: binary anomaly labels

This makes the converted dataset readable by most GAD baselines that expect the
Amazon/Reddit/YelpChi-style .mat format.
"""

import argparse
import os
import pickle
import sys
from typing import Any, Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def safe_pickle_load(path: str) -> Any:
    """Load a pickle file and give a clear error for truncated downloads."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"Failed to unpickle {path}. The file is probably incomplete/truncated. "
            f"Please re-download it. Original error: {repr(e)}"
        ) from e
    except EOFError as e:
        raise RuntimeError(
            f"Failed to unpickle {path}. The file ended unexpectedly. "
            f"Please re-download it. Original error: {repr(e)}"
        ) from e


def to_csr(x: Any, name: str = "matrix") -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr()
    return sp.csr_matrix(x)


def to_2d_feature_matrix(x: Any, name: str) -> sp.csr_matrix:
    """Convert dense/sparse feature object to CSR, keeping memory use modest."""
    if sp.issparse(x):
        feat = x.tocsr()
    else:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        feat = sp.csr_matrix(arr)
    if feat.ndim != 2:
        raise ValueError(f"{name} should be 2-D, got shape {feat.shape}")
    return feat


def normalize_binary_labels(labels: Any) -> np.ndarray:
    y = np.asarray(labels).reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)
    # Some files may store bool, {-1,1}, {0,1}, or class ids. Treat >0 as anomaly.
    y = np.array([1 if v > 0 else 0 for v in y], dtype=np.int64)
    return y


def remove_self_loops_and_binarize(adj: sp.csr_matrix, binarize: bool = True) -> sp.csr_matrix:
    adj = adj.tocsr(copy=True)
    adj.setdiag(0)
    adj.eliminate_zeros()
    if binarize:
        adj.data[:] = 1.0
    return adj.tocsr()


def make_symmetric(adj: sp.csr_matrix, binarize: bool = True) -> sp.csr_matrix:
    adj = adj.tocsr()
    adj = adj.maximum(adj.T)
    if binarize:
        adj.data[:] = 1.0
    return adj.tocsr()


def generate_u2u_from_u2p(u2p: sp.csr_matrix, binarize: bool = True) -> sp.csr_matrix:
    """Generate user-user graph from user-post/hashtag graph."""
    u2p = u2p.tocsr()
    # Binarize incidence before projection to reduce the influence of repeated counts.
    u2p_bin = u2p.copy()
    u2p_bin.data[:] = 1.0
    u2u = u2p_bin @ u2p_bin.T
    u2u = remove_self_loops_and_binarize(u2u, binarize=binarize)
    return u2u


def resolve_file(input_dir: str, filename: str, required: bool = True) -> Optional[str]:
    path = os.path.join(input_dir, filename)
    if os.path.exists(path):
        return path
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return None


def load_weibo_components(
    input_dir: str,
    feature_mode: str = "all",
    graph_mode: str = "auto",
    binarize_graph: bool = True,
    make_undirected: bool = True,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """Load Weibo pickle files and return (adj, features, labels)."""
    label_path = resolve_file(input_dir, "weibo_s_labels_u.pkl")
    labels = normalize_binary_labels(safe_pickle_load(label_path))
    n_users = labels.shape[0]

    # Features
    bow_path = resolve_file(input_dir, "weibo_s_features_bow.pkl", required=(feature_mode in ["all", "bow"]))
    loc_path = resolve_file(input_dir, "weibo_s_features_loc.pkl", required=(feature_mode in ["all", "loc"]))

    feat_parts = []
    if feature_mode in ["all", "loc"]:
        feat_loc = to_2d_feature_matrix(safe_pickle_load(loc_path), "features_loc")
        feat_parts.append(feat_loc)
    if feature_mode in ["all", "bow"]:
        feat_bow = to_2d_feature_matrix(safe_pickle_load(bow_path), "features_bow")
        feat_parts.append(feat_bow)

    if not feat_parts:
        raise ValueError(f"Invalid feature_mode={feature_mode}. Use all, bow, or loc.")

    features = sp.hstack(feat_parts, format="csr") if len(feat_parts) > 1 else feat_parts[0].tocsr()

    if features.shape[0] != n_users:
        raise ValueError(
            f"Feature-label mismatch: features has {features.shape[0]} rows, labels has {n_users}."
        )

    # Graph
    u2u_path = resolve_file(input_dir, "weibo_s_graph_u2u.pkl", required=False)
    u2p_path = resolve_file(input_dir, "weibo_s_graph_u2p.pkl", required=False)

    adj = None
    if graph_mode in ["auto", "u2u"] and u2u_path is not None:
        try:
            adj = to_csr(safe_pickle_load(u2u_path), "graph_u2u")
            print(f"Loaded graph_u2u from {u2u_path}")
        except Exception as e:
            if graph_mode == "u2u":
                raise
            print(f"Warning: failed to load graph_u2u ({e}). Will try graph_u2p projection.")
            adj = None

    if adj is None:
        if graph_mode == "u2u":
            raise FileNotFoundError("graph_mode='u2u' but weibo_s_graph_u2u.pkl is missing or broken.")
        if u2p_path is None:
            raise FileNotFoundError(
                "No usable graph found. Need weibo_s_graph_u2u.pkl or weibo_s_graph_u2p.pkl."
            )
        u2p = to_csr(safe_pickle_load(u2p_path), "graph_u2p")
        print(f"Generating user-user graph from graph_u2p: {u2p.shape}")
        adj = generate_u2u_from_u2p(u2p, binarize=binarize_graph)

    if adj.shape[0] != n_users or adj.shape[1] != n_users:
        raise ValueError(
            f"Graph-label mismatch: adj shape={adj.shape}, labels has {n_users}. "
            "For homogeneous GAD baselines, use user-user graph_u2u or projected graph_u2p @ graph_u2p.T."
        )

    adj = remove_self_loops_and_binarize(adj, binarize=binarize_graph)
    if make_undirected:
        adj = make_symmetric(adj, binarize=binarize_graph)

    return adj.tocsr(), features.tocsr(), labels


def save_mat(
    output_path: str,
    adj: sp.csr_matrix,
    features: sp.csr_matrix,
    labels: np.ndarray,
    save_aliases: bool = True,
    dense_features: bool = False,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    feat_to_save: Any = features
    if dense_features:
        feat_to_save = features.toarray().astype(np.float32)
    else:
        feat_to_save = features.astype(np.float32).tocsr()

    label_col = labels.reshape(-1, 1).astype(np.int64)

    data = {
        "Network": adj.astype(np.float32).tocsr(),
        "Attributes": feat_to_save,
        "Label": label_col,
    }

    # Add common aliases used by different baselines.
    if save_aliases:
        data.update({
            "A": data["Network"],
            "net": data["Network"],
            "homo": data["Network"],
            "X": data["Attributes"],
            "features": data["Attributes"],
            "gnd": data["Label"],
            "labels": data["Label"],
        })

    sio.savemat(output_path, data, do_compression=True)


def print_summary(adj: sp.csr_matrix, features: sp.csr_matrix, labels: np.ndarray) -> None:
    n = labels.shape[0]
    e = adj.nnz
    undirected_edges = e // 2 if (adj != adj.T).nnz == 0 else e
    print("\n========== Weibo MAT Summary ==========")
    print(f"nodes: {n}")
    print(f"adj shape: {adj.shape}")
    print(f"directed nnz entries: {e}")
    print(f"approx undirected edges: {undirected_edges}")
    print(f"feature shape: {features.shape}")
    print(f"anomalies: {int(labels.sum())} ({labels.mean() * 100:.2f}%)")
    print("keys saved: Network, Attributes, Label, plus aliases A/net/homo, X/features, gnd/labels")


def check_files(input_dir: str) -> None:
    print(f"Checking pickle files under: {input_dir}")
    candidates = [
        "weibo_s_graph_u2u.pkl",
        "weibo_s_graph_u2p.pkl",
        "weibo_s_features_bow.pkl",
        "weibo_s_features_loc.pkl",
        "weibo_s_labels_u.pkl",
    ]
    for name in candidates:
        path = os.path.join(input_dir, name)
        print(f"\n{name}")
        if not os.path.exists(path):
            print("  MISSING")
            continue
        print(f"  size: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
        try:
            obj = safe_pickle_load(path)
            print(f"  OK type: {type(obj)}")
            if hasattr(obj, "shape"):
                print(f"  shape: {obj.shape}")
            elif hasattr(obj, "__len__"):
                print(f"  len: {len(obj)}")
        except Exception as e:
            print(f"  BROKEN: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Weibo_s pickle dataset to common GAD .mat format.")
    parser.add_argument("--input_dir", type=str, default="../datasets/weibo",
                        help="Directory containing weibo_s_*.pkl files.")
    parser.add_argument("--output", type=str, default="../datasets/Weibo.mat",
                        help="Output .mat path.")
    parser.add_argument("--feature", type=str, default="all", choices=["all", "bow", "loc"],
                        help="Feature mode: all=loc+bow, bow=BOW only, loc=location only.")
    parser.add_argument("--graph", type=str, default="auto", choices=["auto", "u2u", "u2p"],
                        help="Graph source: auto uses u2u if valid, otherwise projects u2p; u2p always projects u2p @ u2p.T.")
    parser.add_argument("--no_binarize", action="store_true",
                        help="Keep original graph weights instead of binarizing edges.")
    parser.add_argument("--directed", action="store_true",
                        help="Do not symmetrize adjacency.")
    parser.add_argument("--dense_features", action="store_true",
                        help="Save Attributes as dense matrix. Default saves sparse CSR to reduce file size.")
    parser.add_argument("--no_aliases", action="store_true",
                        help="Only save Network/Attributes/Label without aliases.")
    parser.add_argument("--check", action="store_true",
                        help="Only check whether pickle files can be loaded; do not convert.")
    args = parser.parse_args()

    if args.check:
        check_files(args.input_dir)
        return

    adj, features, labels = load_weibo_components(
        input_dir=args.input_dir,
        feature_mode=args.feature,
        graph_mode=args.graph,
        binarize_graph=not args.no_binarize,
        make_undirected=not args.directed,
    )
    print_summary(adj, features, labels)
    save_mat(
        output_path=args.output,
        adj=adj,
        features=features,
        labels=labels,
        save_aliases=not args.no_aliases,
        dense_features=args.dense_features,
    )
    print(f"\nSaved .mat to: {args.output}")


if __name__ == "__main__":
    main()
