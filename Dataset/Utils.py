from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np 
from typing import Dict, List

def prepare_data_range(train_term_df, embed_ids: list, embeds_np: np.ndarray, top_range=[0,100]):
    """Prepare dataset structures using numpy arrays for embeddings to save memory.

    Args:
        train_term_df (pd.DataFrame): dataframe with columns ['EntryID', 'term']
        embed_ids (array-like): array of EntryIDs aligned with rows of `embeds_np`
        embeds_np (np.ndarray): embedding array with shape (N, ...) where N == len(embed_ids)
        top_range (list): slice indices into sorted term frequencies to select top terms

    Returns:
        dict: same structure as before but with 'embeds' as a list of numpy arrays (no pandas DataFrame used)
    """

    term_counts = train_term_df['term'].value_counts()
    top_terms = term_counts[top_range[0]: top_range[1]].index.tolist()

    # Filter training terms to only the selected top terms
    filtered_df = train_term_df[train_term_df['term'].isin(top_terms)]
    # map EntryID -> list of terms
    entry_terms = filtered_df.groupby('EntryID')['term'].apply(list).to_dict()

    # Build id -> index map for the embedding array
    id_list = list(embed_ids)
    id_to_idx = {id_list[i]: i for i in range(len(id_list))}

    # Keep only entries that are present in both the term annotations and the embedding ids
    valid_entries = [eid for eid in id_list if eid in entry_terms]

    # Extract embeddings and labels in the order of valid_entries
    valid_embeds = [embeds_np[id_to_idx[eid]] for eid in valid_entries]
    labels_list = [entry_terms[eid] for eid in valid_entries]

    mlb = MultiLabelBinarizer(classes=top_terms)
    labels_binary = mlb.fit_transform(labels_list)

    return {
        'entries': valid_entries,
        'embeds': valid_embeds,
        'labels': labels_binary,
        'mlb': mlb,
        'top_terms': top_terms,
        'num_classes': len(top_terms)
    }

def get_class_frequencies_from_dataframe(train_term_df, top_terms):

    from collections import Counter

    # Count frequency of each term
    term_counts = Counter(train_term_df['term'])

    # Get frequencies for top_terms in order
    class_frequencies = np.array([term_counts[term] for term in top_terms])

    return class_frequencies


def read_fasta(path: str) -> Dict[str, str]:
    seqs = {}
    with open(path, "r") as f:
        pid = None; seq_parts = []
        for line in f:
            line=line.strip()
            if line.startswith(">"):
                if pid: seqs[pid] = "".join(seq_parts)
                header=line[1:].split()[0]
                if "|" in header:
                    parts=header.split("|"); pid = parts[1] if len(parts)>=2 else header
                else:
                    pid = header
                seq_parts=[]
            else:
                seq_parts.append(line.strip())
        if pid: seqs[pid] = "".join(seq_parts)
    print(f"[io] Read {len(seqs)} sequences from {path}")
    return seqs


def load_and_merge_embeddings(embeds_paths, ids_paths=None, ids_path=None):
    """Load multiple embedding .npy files and merge them aligned by EntryID.

    Args:
        embeds_paths (list[str]): list of paths to embedding .npy files. Each file must be a numpy array
            with shape (N, ...) where first dimension indexes samples.
        ids_paths (list[str] or None): optional list of paths to ids .npy files matching each embed file.
        ids_path (str or None): optional single ids file used for all embed files (if ids_paths not provided).

    Returns:
        tuple: (merged_ids, merged_embeds)
            merged_ids: numpy array of EntryIDs that are common across all files (order preserved from first ids array).
            merged_embeds: numpy array of shape (len(merged_ids), total_dim) with concatenated features.

    The function will keep only EntryIDs present in all files (intersection). If ids_paths is provided, the
    ids arrays across files must match exactly (same order) or the function will align by EntryID.
    """
    import numpy as _np

    if not isinstance(embeds_paths, (list, tuple)):
        raise ValueError("embeds_paths must be a list of file paths")

    loaded_embeds = [_np.load(p) for p in embeds_paths]

    # load ids
    if ids_paths is not None:
        if not isinstance(ids_paths, (list, tuple)):
            raise ValueError("ids_paths must be a list when providing multiple embedding files")
        if len(ids_paths) != len(loaded_embeds):
            raise ValueError("Length of ids_paths must match embeds_paths")
        loaded_ids = [_np.load(p) for p in ids_paths]
    else:
        if ids_path is None:
            raise ValueError("Either ids_paths or ids_path must be provided to align embeddings")
        base_ids = _np.load(ids_path)
        loaded_ids = [base_ids for _ in loaded_embeds]

    # Build mapping from id -> embedding for each file (flatten per-sample to 1D feature vector)
    per_file_maps = []
    for arr, ids in zip(loaded_embeds, loaded_ids):
        n = arr.shape[0]
        if ids.shape[0] != n:
            raise ValueError("Embedding file and ids file have different first-dimension sizes")
        # flatten remaining dims to 2D (N, -1)
        flat = arr.reshape(n, -1)
        per_file_maps.append(dict(zip(ids.tolist(), [flat[i] for i in range(n)])))

    # compute intersection of ids across all files
    id_sets = [set(m.keys()) for m in per_file_maps]
    common_ids = sorted(list(set.intersection(*id_sets)), key=lambda x: list(per_file_maps[0].keys()).index(x) if x in per_file_maps[0] else 0)

    # Build merged embeddings in the order of common_ids
    merged_list = []
    for eid in common_ids:
        parts = [_np.asarray(m[eid]).reshape(-1) for m in per_file_maps]
        merged = _np.concatenate(parts, axis=0)
        merged_list.append(merged)

    if len(merged_list) == 0:
        return _np.array([], dtype=object), _np.zeros((0, 0))

    merged_embeds = _np.stack(merged_list, axis=0)
    merged_ids = _np.array(common_ids)
    return merged_ids, merged_embeds