import numpy as np
import pandas as pd
import pickle
import obonet


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
    