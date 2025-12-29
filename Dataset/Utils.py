from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np 
from typing import Dict, List
import pickle
import obonet

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

    valid_embeds = np.array(valid_embeds)
    valid_entries = np.array(valid_entries)
    labels_binary = np.array(labels_binary)
    
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

def pad_terms_with_neighbors(terms, go_graph, go_embeds, max_terms=256):
    """
    Pad a list of GO terms with neighboring terms from the GO graph.
    
    Args:
        terms: List of GO term IDs
        go_graph: networkx graph of GO ontology
        go_embeds: Dictionary of GO embeddings (to check if term has embedding)
        max_terms: Maximum number of terms to return
    
    Returns:
        List of GO terms padded to max_terms
    """
    padded_terms = list(terms)
    
    if len(padded_terms) >= max_terms:
        return padded_terms[:max_terms]
    
    # Use a set for faster lookup
    terms_set = set(padded_terms)
    
    # Collect neighbors from existing terms
    candidates = set()
    for term in padded_terms:
        if term in go_graph:
            # Get parents (is_a relationships)
            if 'is_a' in go_graph.nodes[term]:
                parents = go_graph.nodes[term].get('is_a', [])
                if isinstance(parents, str):
                    parents = [parents]
                candidates.update(parents)
            
            # Get successors (children) and predecessors (parents)
            candidates.update(go_graph.successors(term))
            candidates.update(go_graph.predecessors(term))
    
    # Remove terms already in the list
    candidates = candidates - terms_set
    
    # Filter candidates to only those with embeddings
    candidates = [c for c in candidates if c in go_embeds]
    
    # Add candidates until we reach max_terms
    for candidate in candidates:
        if len(padded_terms) >= max_terms:
            break
        padded_terms.append(candidate)
        terms_set.add(candidate)
    
    # If still not enough, try neighbors of neighbors
    if len(padded_terms) < max_terms:
        second_level_candidates = set()
        for term in candidates[:100]:  # Limit to avoid too much computation
            if term in go_graph:
                if 'is_a' in go_graph.nodes[term]:
                    parents = go_graph.nodes[term].get('is_a', [])
                    if isinstance(parents, str):
                        parents = [parents]
                    second_level_candidates.update(parents)
                second_level_candidates.update(go_graph.successors(term))
                second_level_candidates.update(go_graph.predecessors(term))
        
        second_level_candidates = second_level_candidates - terms_set
        second_level_candidates = [c for c in second_level_candidates if c in go_embeds]
        
        for candidate in second_level_candidates:
            if len(padded_terms) >= max_terms:
                break
            padded_terms.append(candidate)
    
    return padded_terms

def pad_dataframe_terms(seq_2_terms_df, go_graph, go_embeds, max_terms=256):
    """
    Pad the terms_predicted column in the dataframe in-place using GO graph neighbors.
    
    Args:
        seq_2_terms_df: DataFrame with 'terms_predicted' column
        go_graph: networkx graph of GO ontology
        go_embeds: Dictionary of GO embeddings
        max_terms: Maximum number of terms per row
    """
    print("Padding terms_predicted with GO graph neighbors...")
    seq_2_terms_df['terms_predicted'] = seq_2_terms_df['terms_predicted'].apply(
        lambda terms: pad_terms_with_neighbors(terms, go_graph, go_embeds, max_terms)
    )
    print(f"Padding complete. Average terms per row: {seq_2_terms_df['terms_predicted'].apply(len).mean():.2f}")
    return seq_2_terms_df

def prepare_data(data_paths, max_terms=256, aspect=None):
    
    knn_terms_df = data_paths['knn_terms_df']
    train_terms_df = data_paths['train_terms_df']
    features_embeds_path = data_paths['features_embeds_path']
    features_ids_path = data_paths['features_ids_path']

    go_embeds_paths = data_paths['go_embeds_paths']

    seq_2_terms = pd.read_parquet(knn_terms_df, engine='fastparquet')
    train_terms = pd.read_csv(train_terms_df, sep='\t')

    term_to_aspect = train_terms.groupby('term')['aspect'].first().to_dict()

    go_graph = obonet.read_obo(data_paths['go_obo_path'])
        
    with open(go_embeds_paths, 'rb') as f:
        data = pickle.load(f)
        embeddings_dict = data['embeddings']
        go_ids = data['go_ids']

    # Filter to keep only terms from a specific aspect if aspect is provided
    if aspect is not None:
        seq_2_terms['terms_predicted'] = seq_2_terms['terms_predicted'].apply(
            lambda terms: [t for t in terms if term_to_aspect.get(t) == aspect]
        )
        seq_2_terms['terms_true'] = seq_2_terms['terms_true'].apply(
            lambda terms: [t for t in terms if term_to_aspect.get(t) == aspect]
        )
        # Remove rows where terms_predicted or terms_true is now empty
        seq_2_terms = seq_2_terms[seq_2_terms['terms_predicted'].apply(len) > 0]
        seq_2_terms = seq_2_terms[seq_2_terms['terms_true'].apply(len) > 0]


    features_embeds = np.load(features_embeds_path, allow_pickle=True)
    features_ids = np.load(features_ids_path, allow_pickle=True)

    features_embeds_dict = {feat_id: embed for feat_id, embed in zip(features_ids, features_embeds)}

    # Pad terms_predicted in the dataframe with GO graph neighbors
    seq_2_terms = pad_dataframe_terms(seq_2_terms, go_graph, embeddings_dict, max_terms=max_terms)
    #remove seq_2_terms row for which len(predicted_terms == 0)

    term_lengths = seq_2_terms['terms_predicted'].apply(len)

    #currently only using sequences with 256 terms, need to change later 
    seq_2_terms = seq_2_terms[term_lengths == max_terms]

    train_ids =  pd.DataFrame(features_ids, columns=['qseqid'])
    seq_2_terms = seq_2_terms.merge(train_ids, on='qseqid', how='inner')    

    out = {'seq_2_terms': seq_2_terms,
           'train_terms': train_terms,
           'features_embeds': features_embeds_dict,
           'go_embeds': embeddings_dict,
           'go_graph': go_graph
           }
    return out