"""Graph loading, unlabeled training references and stratified evaluation splits."""
from __future__ import annotations
from pathlib import Path
from collections import deque
import csv
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def unwrap(v):
    while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
        v = v.reshape(-1)[0]
    return v

def pick(m, names, required=True):
    lookup = {str(k).lower(): k for k in m}
    for name in names:
        if name.lower() in lookup:
            return unwrap(m[lookup[name.lower()]])
    if required:
        raise KeyError(f'Missing {names}; available keys: {list(m)}')
    return None

def read_hdf5_mat(path):
    try:
        import h5py
    except ImportError as e:
        raise ImportError('MATLAB v7.3 requires: pip install h5py') from e
    def read(obj, file):
        if isinstance(obj, h5py.Group):
            if {'ir', 'jc', 'data'}.issubset(obj.keys()):
                rows = int(np.asarray(obj.attrs['MATLAB_sparse']).reshape(-1)[0])
                jc = np.asarray(obj['jc']).reshape(-1)
                return sp.csc_matrix((np.asarray(obj['data']).reshape(-1),
                                      np.asarray(obj['ir']).reshape(-1), jc),
                                     shape=(rows, len(jc)-1))
            return {k: read(v, file) for k, v in obj.items() if not k.startswith('#')}
        a = np.asarray(obj)
        if h5py.check_dtype(ref=a.dtype) is not None:
            if a.size != 1:
                raise ValueError('Multiple MATLAB cell references: specify a single graph.')
            return read(file[a.reshape(-1)[0]], file)
        # MATLAB HDF5 stores dimension order reversed.
        return a.T if a.ndim == 2 else a
    with h5py.File(path, 'r') as f:
        return {k: read(v, f) for k, v in f.items() if not k.startswith('#')}

def sparse_component(m, prefix):
    keys = [prefix + '_' + s for s in ('data', 'indices', 'indptr', 'shape')]
    if all(k in m for k in keys):
        d, i, p, shape = [m[k] for k in keys]
        return sp.csr_matrix((d, i, p), shape=tuple(np.asarray(shape).reshape(-1)))
    return None

def load_graph(path, preprocessing='drop_constant'):
    """Return attributes, symmetric weighted CSR adjacency, binary labels.

    MAT vectors [1,N]/[N,1], sparse labels and object-wrapped arrays supported.
    NPZ supports Questions edges, direct adjacency and attr_*/adj_* CSR arrays.
    Extensionless files (including GADBench graphs) are read as DGL binaries.
    Labels must be {0,1}, with -1 optionally denoting unknown. No multiclass
    labels are silently converted into anomalies. Pickled NPZ must be trusted.
    """
    path = Path(path)
    with path.open('rb') as file:
        magic = file.read(8)
    # DGL graph serialization magic; handles YelpChiFull renamed with .mat.
    is_dgl = magic == bytes.fromhex('3fa1b446f04f2edd')
    if (is_dgl or path.suffix.lower() == '.bin' or
            path.suffix == ''):
        try:
            from dgl.data.utils import load_graphs
        except ImportError as e:
            raise ImportError(f'{path.name} is a DGL graph binary (possibly renamed .mat). '
                              'Install DGL compatible with your existing PyTorch environment.') from e
        gs, _ = load_graphs(str(path))
        if len(gs) != 1:
            raise ValueError('Expected one graph in DGL file.')
        g = gs[0]
        src, dst = g.edges()
        x_np = pick(g.ndata, ['feat','feature','features','x']).cpu().numpy()
        y_np = pick(g.ndata, ['label','labels','y']).cpu().numpy()
        # Avoid a second 2 x E stacked int64 edge array.
        adjacency = sp.csr_matrix((np.ones(len(src), np.float32),
            (src.cpu().numpy(), dst.cpu().numpy())), shape=(len(x_np), len(x_np)))
        m = {'features': x_np, 'labels': y_np, 'adj': adjacency}
        del gs, g, src, dst, adjacency
    elif path.suffix.lower() == '.mat':
        try:
            m = sio.loadmat(path)
        except (NotImplementedError, ValueError) as e:
            if '7.3' not in str(e) and 'Unknown mat' not in str(e):
                raise
            m = read_hdf5_mat(path)
    elif path.suffix.lower() in ('.npz', '.npy'):
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.lib.npyio.NpzFile):
            with raw:
                m = {k: unwrap(raw[k]) for k in raw.files}
        else:
            m = unwrap(raw)
            if not isinstance(m, dict):
                raise ValueError('NPY must contain a graph dictionary.')
    else:
        raise ValueError(f'Unsupported format: {path.suffix}')
    # Common MATLAB single struct wrapper.
    public = {k: unwrap(v) for k,v in m.items() if not k.startswith('__')}
    if len(public) == 1:
        v = next(iter(public.values()))
        if isinstance(v, dict):
            m = v
        elif isinstance(v, np.ndarray) and v.dtype.names and v.size == 1:
            m = {k: unwrap(v[k]) for k in v.dtype.names}
    x = sparse_component(m, 'attr')
    if x is None:
        x = pick(m, ['node_features','features','feature','Attributes','attr','X'])
    x = sp.csr_matrix(x, dtype=np.float32) if sp.issparse(x) else np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f'Features must be [N,D], got {x.shape}')
    n = x.shape[0]
    if n < 2:
        raise ValueError(f'Only {n} feature rows; inspect feature orientation, not train/test ratios.')
    raw_y = pick(m, ['node_labels','labels','label','gnd','y'])
    y = raw_y.toarray() if sp.issparse(raw_y) else np.asarray(raw_y)
    y = np.squeeze(y)
    if y.ndim == 2 and y.shape in [(n,2),(2,n)]:
        y = y if y.shape == (n,2) else y.T
        if not np.all((y == 0) | (y == 1)) or not np.all(y.sum(1) == 1):
            raise ValueError('Two-column labels must be binary one-hot, not probabilities.')
        y = y.argmax(1)
    if y.size != n or y.ndim != 1:
        raise ValueError(f'Label shape {y.shape} does not match feature node count {n}.')
    if not np.isin(y, [-1,0,1]).all():
        raise ValueError(f'Expected normal=0, anomaly=1, unknown=-1; got {np.unique(y)}')
    y = y.astype(np.int64)
    a = sparse_component(m, 'adj')
    if a is None:
        a = pick(m, ['Network','net','homo','adjacency','adj','A'], required=False)
    if a is not None:
        a = sp.csr_matrix(unwrap(a), dtype=np.float32)
        if a.shape != (n,n):
            raise ValueError(f'Adjacency {a.shape} != ({n},{n})')
    else:
        edges = np.asarray(pick(m, ['edge_index','edges']))
        if edges.ndim != 2:
            raise ValueError(f'Edges must be [2,E] or [E,2], got {edges.shape}')
        if edges.shape[0] != 2:
            if edges.shape[1] != 2:
                raise ValueError(f'Invalid edges: {edges.shape}')
            edges = edges.T
        if not np.isfinite(edges).all() or not np.equal(edges, np.floor(edges)).all():
            raise ValueError('Edge indices must be finite integers.')
        edges = edges.astype(np.int64)
        if edges.size and (edges.min() < 0 or edges.max() >= n):
            raise ValueError('Edge indices out of range. Use zero-based node indices.')
        w = pick(m, ['edge_weight','edge_weights'], required=False)
        w = np.ones(edges.shape[1], np.float32) if w is None else np.asarray(w, np.float32).reshape(-1)
        a = sp.csr_matrix((w, (edges[0], edges[1])), shape=(n,n))
    if not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError('Adjacency weights must be finite and nonnegative.')
    a.sum_duplicates()
    # Maximum symmetrization preserves binary adjacency without doubling edges.
    a = a.maximum(a.T).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    a.sort_indices()
    data = x.data if sp.issparse(x) else x
    if not np.isfinite(data).all():
        raise ValueError('Features contain NaN/Inf; clean them before training.')
    # A nonzero constant column survives sparse StandardScaler(with_mean=False).
    # Drop it before scaling: otherwise its magnitude can dominate graph energy.
    if preprocessing not in ('legacy', 'drop_constant'):
        raise ValueError('preprocessing must be legacy or drop_constant')
    if preprocessing == 'drop_constant':
        if sp.issparse(x):
            from sklearn.utils.sparsefuncs import mean_variance_axis
            _, feature_var = mean_variance_axis(x.astype(np.float64), axis=0)
        else:
            feature_var = np.var(x, axis=0, dtype=np.float64)
        keep = feature_var > 0
        print(f'  Drop constant columns (zero-based): {np.flatnonzero(~keep).tolist()}', flush=True)
        if not keep.any():
            raise ValueError('All attribute columns are constant; no attribute signal remains.')
        x = x[:, keep]
    # Transductive preprocessing: all observed attributes, no label information.
    x = StandardScaler(with_mean=not sp.issparse(x)).fit_transform(x).astype(np.float32)
    if sp.issparse(x):
        x = x.tocsr()
    print(f'Loaded {path.name}: X={x.shape}, directed_edges={a.nnz}, '
          f'normal={(y==0).sum()}, anomaly={(y==1).sum()}, unknown={(y<0).sum()}', flush=True)
    return x,a,y

def make_split(y, early_stop, seed):
    """Stratify known labels for evaluation only; False gives train:test = 8:2.

    Unknown nodes join the training reference, so 8:2 applies to known nodes.
    The model and candidate sampler never receive these stratification labels.
    """
    known = np.flatnonzero(y >= 0)
    classes,counts = np.unique(y[known], return_counts=True)
    if len(classes) != 2 or counts.min() < (3 if early_stop else 2):
        raise ValueError('Each class needs >=3 nodes for 7:1:2 or >=2 for 8:2.')
    try:
        trainval,test = train_test_split(known, test_size=.2, stratify=y[known], random_state=seed)
        if early_stop:
            train,val = train_test_split(trainval, test_size=.125,
                                        stratify=y[trainval], random_state=seed+1)
        else:
            train,val = trainval,np.empty(0,dtype=np.int64)
    except ValueError as e:
        raise ValueError(f'Cannot stratify node labels (shape={y.shape}, counts={counts.tolist()}): {e}') from e
    for name, ids in [('train',train),('validation',val),('test',test)]:
        if name=='validation' and not early_stop:
            continue
        if len(np.unique(y[ids]))<2:
            raise ValueError(f'{name} split lacks one class at the requested ratios; '
                             f'original class counts={counts.tolist()}. More minority nodes are needed.')
    # Unknown nodes may participate in unsupervised training, never evaluation.
    train = np.sort(np.r_[train,np.flatnonzero(y<0)])
    return train,np.sort(val),np.sort(test)

def reference_ranks(scores, reference):
    """Midrank percentile against train scores, preserving exact ties."""
    ref = np.sort(scores[reference])
    if len(ref) < 2:
        raise ValueError('Need >=2 training reference nodes.')
    lo = np.searchsorted(ref,scores,side='left')
    hi = np.searchsorted(ref,scores,side='right')
    return np.clip((lo+hi-1)/(2*(len(ref)-1)),0,1).astype(np.float32)

def sample_pairs(rank, count, gap, rng):
    if len(rank)<2 or np.ptp(rank)<gap or np.ptp(rank)==0:
        empty=np.empty(0,dtype=np.int64)
        return empty,empty,np.empty(0,np.float32)
    hs,ls,ws=[],[],[]
    total=0
    for _ in range(20):
        size=max(256,3*(count-total))
        i=rng.integers(len(rank),size=size)
        j=rng.integers(len(rank),size=size)
        d=rank[i]-rank[j]
        valid=(np.abs(d)>=gap)&(d!=0)&(i!=j)
        i,j,d=i[valid],j[valid],d[valid]
        hs.append(np.where(d>0,i,j)); ls.append(np.where(d>0,j,i))
        ws.append(np.abs(d)); total+=len(d)
        if total>=count: break
    return np.concatenate(hs)[:count],np.concatenate(ls)[:count],np.concatenate(ws)[:count]

def topology_partitions(a, target):
    """Disjoint BFS cores; spectral_batches later adds exact K-hop halos."""
    seen=np.zeros(a.shape[0],bool)
    cores=[]; current=[]
    for root in range(a.shape[0]):
        if seen[root]: continue
        queue=deque([root]); seen[root]=True
        while queue:
            u=queue.popleft(); current.append(u)
            if len(current)==target:
                cores.append(np.asarray(current,dtype=np.int64)); current=[]
            nbr=a.indices[a.indptr[u]:a.indptr[u+1]]
            new=nbr[~seen[nbr]]
            seen[new]=True; queue.extend(new.tolist())
    if current: cores.append(np.asarray(current,dtype=np.int64))
    return cores

def metrics(y,s,ids):
    if len(ids)==0 or len(np.unique(y[ids]))<2:
        return {'auroc':float('nan'),'auprc':float('nan')}
    return {'auroc':float(roc_auc_score(y[ids],s[ids])),
            'auprc':float(average_precision_score(y[ids],s[ids]))}

def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def discover(root,names):
    root=Path(root)
    if not root.is_dir(): raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    files=sorted(p for p in root.iterdir() if p.is_file() and
                 (p.suffix.lower() in ['.mat','.npz','.npy','.bin'] or
                  p.suffix==''))
    if names==['all']: return files
    selected=[]
    for name in names:
        match=[p for p in files if p.name.lower()==name.lower()]
        if not match:
            match=[p for p in files if p.stem.lower()==name.lower()]
        if len(match)!=1: raise ValueError(f'{name}: expected one matching data file, found {match}')
        selected.append(match[0])
    return selected
