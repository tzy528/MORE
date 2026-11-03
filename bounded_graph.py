"""Deterministic disk-backed subgraphs and CPU, feature-chunked teachers.

The local operator is an approximation when neighborhoods/edges are truncated.
Weights ALWAYS use the original full-graph degrees. No dense N x N arrays.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp


def dense_columns(x, rows, start, end):
    # Slice sparse columns BEFORE rows/densification: never densify N x D.
    b = x[:, start:end][rows] if sp.issparse(x) else x[rows, start:end]
    return np.asarray(b.toarray() if sp.issparse(b) else b, dtype=np.float32)


def grow_halo(core, a, hops, cap, fanout, rng):
    chosen = set(map(int, core))
    frontier = list(map(int, core))
    sampled = False
    for _ in range(hops):
        new = []
        for u in frontier:
            nbr = a.indices[a.indptr[u]:a.indptr[u + 1]]
            if len(nbr) > fanout:
                nbr = nbr[rng.choice(len(nbr), fanout, replace=False)]
                sampled = True
            for v in nbr:
                v = int(v)
                if v not in chosen:
                    if len(chosen) >= cap:
                        sampled = True
                        continue
                    chosen.add(v)
                    new.append(v)
        frontier = new
        if not frontier:
            break
    return np.asarray(sorted(chosen), dtype=np.int64), sampled


def local_operator(a, halo, inv_degree, edge_cap, rng):
    """Stream CSR rows; never materialize a[halo, :] or a dense adjacency.

    Per-row selection followed by symmetric union guarantees nnz <= edge_cap.
    Self loops were removed by the loader. CPU temporary memory is O(max degree).
    """
    quota = max(1, edge_cap // (2 * len(halo)))
    rows, cols, vals = [], [], []
    cut = False
    for local, u in enumerate(halo):
        begin, end = a.indptr[u:u+2]
        nbr = a.indices[begin:end]
        loc = np.searchsorted(halo, nbr)
        valid = loc < len(halo)
        ix = np.flatnonzero(valid)
        ix = ix[halo[loc[ix]] == nbr[ix]]
        if len(ix) > quota:
            ix = np.sort(rng.choice(ix, quota, replace=False))
            cut = True
        rows.extend([local] * len(ix))
        cols.extend(loc[ix].tolist())
        vals.extend((a.data[begin:end][ix] * inv_degree[u] * inv_degree[nbr[ix]]).tolist())
    s = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                      shape=(len(halo), len(halo)), dtype=np.float32)
    s = s.maximum(s.T).tocsr()
    s.sort_indices()
    if s.nnz > edge_cap:
        raise AssertionError('Edge budget exceeded')
    return -s, cut


class GraphPlan:
    """One partition is held in RAM/GPU at a time; caches are reproducible.

    New output directories are required: stale caches are never silently reused.
    """
    def __init__(self, a, feature_dim, args, folder):
        from graph_io import topology_partitions
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.n = a.shape[0]
        input_cap = int(args.feature_mb * 1024**2 // (4 * max(1, feature_dim)))
        cap = min(args.halo_nodes, input_cap, args.halo_edges // 2)
        if cap < 3:
            raise ValueError('Feature/halo/edge budget permits fewer than three nodes')
        core_cap = min(args.core_nodes, cap)
        degree = np.asarray(a.sum(1)).ravel()
        inv = np.zeros_like(degree)
        np.divide(1, np.sqrt(degree), out=inv, where=degree > 0)
        exact = self.n <= cap and a.nnz <= args.halo_edges
        cores = [np.arange(self.n)] if exact else topology_partitions(a, core_cap)
        self.paths = []
        reports = []
        for part, core in enumerate(cores):
            core = np.sort(core)
            rng = np.random.default_rng(args.partition_seed + part * 1009)
            if exact:
                halo, sampled = np.arange(self.n), False
                # Full small-graph operator, without per-row sampling.
                op = -(sp.diags(inv) @ a @ sp.diags(inv)).tocsr()
                cut = False
            else:
                halo, sampled = grow_halo(core, a, args.order, cap, args.fanout, rng)
                op, cut = local_operator(a, halo, inv, args.halo_edges, rng)
            core_pos = np.searchsorted(halo, core)
            file = self.folder / f'part_{part:06d}.npz'
            np.savez(file, core=core, halo=halo, core_pos=core_pos,
                     data=op.data, indices=op.indices, indptr=op.indptr,
                     shape=np.asarray(op.shape))
            self.paths.append(file)
            full_mass = np.asarray([inv[u] * np.dot(a.data[a.indptr[u]:a.indptr[u+1]],
                inv[a.indices[a.indptr[u]:a.indptr[u+1]]]) for u in core])
            local_mass = -np.asarray(op.sum(1)).ravel()[core_pos]
            retained = np.divide(local_mass, full_mass, out=np.ones_like(full_mass), where=full_mass>0)
            reports.append(dict(part=part, core_nodes=len(core), halo_nodes=len(halo),
                                edges=op.nnz, sampled_halo=sampled, sampled_edges=cut,
                                first_hop_weight_retained_mean=float(retained.mean()),
                                first_hop_weight_retained_min=float(retained.min())))
            if part % 100 == 0:
                print(f'  graph cache {part+1}/{len(cores)}: nodes={len(halo)} edges={op.nnz}', flush=True)
        self.report = dict(exact_small_graph=exact, parts=len(cores), effective_halo_cap=cap,
                           max_nodes=max(r['halo_nodes'] for r in reports),
                           max_edges=max(r['edges'] for r in reports),
                           partition_seed=args.partition_seed, batches=reports)
        (self.folder / 'plan.json').write_text(json.dumps(self.report, indent=2), encoding='utf-8')

    def load(self, part):
        with np.load(self.paths[part]) as f:
            op = sp.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape']))
            return f['core'], f['halo'], f['core_pos'], op

    def raw_energy(self, z, order, feature_chunk):
        result = np.empty((self.n, order), np.float32)
        for part in range(len(self.paths)):
            core, halo, pos, op = self.load(part)
            accum = np.zeros((len(core), order), np.float64)
            for start in range(0, z.shape[1], feature_chunk):
                prev2 = dense_columns(z, halo, start, start + feature_chunk)
                prev1 = op @ prev2
                for k in range(order):
                    if k == 0:
                        response = prev1
                    else:
                        response = 2 * (op @ prev1) - prev2
                        prev2, prev1 = prev1, response
                    r = response[pos]
                    accum[:, k] += np.einsum('ij,ij->i', r, r, dtype=np.float64)
            result[core] = np.log1p(accum).astype(np.float32)
        return result


def calibration(raw, reference, clip, floor):
    center = np.median(raw[reference], axis=0)
    mad = np.median(np.abs(raw[reference] - center), axis=0)
    scale = np.maximum(mad, floor)
    signed = (raw - center) / scale
    score = np.minimum(np.abs(signed).max(1), clip)
    return dict(center=center, mad=mad, scale=scale, signed=signed, score=score)


def exact_energy_cpu(z, a, order, feature_chunk):
    """Optional full-graph reference to quantify subgraph approximation error."""
    degree = np.asarray(a.sum(1)).ravel()
    inv = np.zeros_like(degree)
    np.divide(1, np.sqrt(degree), out=inv, where=degree>0)
    def apply(v):
        return -inv[:, None] * (a @ (inv[:, None] * v))
    accum = np.zeros((a.shape[0], order), np.float64)
    for start in range(0, z.shape[1], feature_chunk):
        p2 = dense_columns(z, slice(None), start, start+feature_chunk)
        p1 = apply(p2)
        for k in range(order):
            if k == 0:
                response = p1
            else:
                response = 2*apply(p1)-p2
                p2, p1 = p1, response
            accum[:, k] += np.einsum('ij,ij->i', response, response, dtype=np.float64)
    return np.log1p(accum).astype(np.float32)
