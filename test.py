import pandas as pd
import numpy as np
import os
import torch
import json
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from Model.Query2Label_pl import Query2Label_pl
from Dataset.EmbeddingsDataset import simple_collate, PrefetchLoaderWithRawFeatures
from Dataset.Utils import pad_with_random_terms


class EmbeddingsTestDataset(Dataset):
    """Dataset for testing that doesn't require 'terms_true' column."""
    def __init__(self, 
                 data, 
                 max_go_embeds = 256,  
                 oversample_indices=None
                ):
        
        self.data = data
        self.max_go_embeds = max_go_embeds
        self.oversample_indices = oversample_indices if oversample_indices is not None else list(range(len(self.data['seq_2_terms'])))
        self.mask_embed = np.zeros(next(iter(self.data['go_embeds'].values())).shape, dtype=np.float32)

    def __len__(self):
        return len(self.oversample_indices)         

    def __getitem__(self, idx):
        sample_idx = self.oversample_indices[idx]

        row = self.data['seq_2_terms'].iloc[sample_idx]
        qseqid = row['qseqid']

        plm_embed = self.data['plm_embeds'][qseqid]
        
        # Get BLM embeddings (PMIDs associated with this protein)
        pmids = self.data['prot_2_pmid'].get(qseqid, [])
        blm_embeds = [self.data['pmid_2_embed'][pmid] for pmid in pmids if pmid in self.data['pmid_2_embed']]
        if blm_embeds:
            blm_embeds = np.array(blm_embeds, dtype=np.float32)
        else:
            blm_embeds = np.zeros((0, next(iter(self.data['pmid_2_embed'].values())).shape[0]), dtype=np.float32)
        
        predicted_terms = row['terms_predicted']
        
        # Check if 'terms_true' exists, otherwise create dummy labels
        if 'terms_true' in row and row['terms_true'] is not None:
            true_terms_set = set(row['terms_true'])
            label = np.array([term in true_terms_set for term in predicted_terms], dtype=np.float32)
            true_terms = row['terms_true']
        else:
            # No ground truth available - create dummy zero labels
            label = np.zeros(len(predicted_terms), dtype=np.float32)
            true_terms = []
        
        # Get GO embeddings
        valid_terms = predicted_terms
        go_embeds = np.array([self.data['go_embeds'].get(term, self.mask_embed) for term in valid_terms])
        
        return {
            'entryID'   : qseqid,
            'plm_embed' : plm_embed,
            'blm_embeds': blm_embeds,
            'go_embed'  : go_embeds,
            'label'     : label,
            'predicted_terms': valid_terms,
            'true_terms': true_terms
        }


def prepare_data_test(data_paths, max_terms=256, aspect=None):
    
    seq_2_terms_df    = data_paths['seq_2_terms_df']
    plm_features_path = data_paths['plm_features_path']
    prot_2_pmid_path  = data_paths['prot_2_pmid_path']
    pmid_2_embed_path = data_paths['pmid_2_embed_path']
    go_term_to_aspect_path = data_paths.get('go_term_to_aspect_path', '/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/go_term_to_aspect.npy')

    go_embeds_paths = data_paths['go_embeds_paths']

    seq_2_terms = pd.read_parquet(seq_2_terms_df, engine='fastparquet')

    print("Loading PLM features...")
    plm_features = np.load(plm_features_path, allow_pickle=True).item()
    
    print("Loading BLM data (protein to PMID mapping and PMID embeddings)...")
    prot_2_pmid = np.load(prot_2_pmid_path, allow_pickle=True).item()
    pmid_2_embed = np.load(pmid_2_embed_path, allow_pickle=True).item()
    
    print("Loading GO term to aspect mapping...")
    go_term_to_aspect = np.load(go_term_to_aspect_path, allow_pickle=True).item()
    
    print(f"Loaded {len(go_term_to_aspect)} GO terms with aspect mappings")
        
    with open(go_embeds_paths, 'rb') as f:
        data = pickle.load(f)
        embeddings_dict = data['embeddings']
        go_ids = data['go_ids']
    
    # Filter to keep only terms from a specific aspect if aspect is provided
    print(f'Filtering by aspect: {aspect}')
    if aspect is not None:
        seq_2_terms['terms_predicted'] = seq_2_terms['terms_predicted'].apply(
            lambda terms: [t for t in terms if go_term_to_aspect.get(t) == aspect]
        )
        # Remove rows where terms_predicted is now empty
        seq_2_terms = seq_2_terms[seq_2_terms['terms_predicted'].apply(len) > 0]
        term_lengths = seq_2_terms['terms_predicted'].apply(len)
        print(f"After aspect filtering -  Mean: {term_lengths.mean():.2f}, Min: {term_lengths.min()}, Max: {term_lengths.max()}")
        print(f"After aspect filtering: {len(seq_2_terms)} sequences")
        print(f"After filtering ")
        # Pad terms with random terms from the same aspect
        print(f"Padding terms_predicted with random terms from aspect {aspect}...")
        # Get all terms from this aspect that have embeddings
        aspect_terms = [term for term, asp in go_term_to_aspect.items() if asp == aspect and term in embeddings_dict]
        all_aspect_terms = np.array(aspect_terms)
        
        tqdm.pandas(desc=f"Padding with random {aspect} terms")
        seq_2_terms['terms_predicted'] = seq_2_terms['terms_predicted'].progress_apply(
            lambda terms: pad_with_random_terms(terms, max_terms, all_aspect_terms)
        )
        
        # Verify padding
        term_lengths_after = seq_2_terms['terms_predicted'].apply(len)
        print(f"After padding - Min: {term_lengths_after.min()}, Max: {term_lengths_after.max()}, Mean: {term_lengths_after.mean():.2f}")

    print("Filtering sequences by term lengths and PLM features availability...")
    # Filter sequences by PLM features availability
    available_proteins = set(plm_features.keys())
    #remove seq_2_terms row for which len(predicted_terms == 0)

    term_lengths = seq_2_terms['terms_predicted'].apply(len)

    #currently only using sequences with 256 terms, need to change later 
    seq_2_terms = seq_2_terms[term_lengths == max_terms]

    # Filter by PLM features availability
    seq_2_terms = seq_2_terms[seq_2_terms['qseqid'].isin(available_proteins)]
    
    print(f"After filtering: {len(seq_2_terms)} sequences with both PLM features and GO terms")

    out = {'seq_2_terms': seq_2_terms,
           'plm_embeds': plm_features,
           'prot_2_pmid': prot_2_pmid,
           'pmid_2_embed': pmid_2_embed,
           'go_embeds': embeddings_dict
        }  

    return out


def load_model_and_tokenizer(model_dir, device):
    """Load model from checkpoint directory."""
    print(f"Loading model from {model_dir}...")
    
    # Load configs
    with open(os.path.join(model_dir, "configs.json"), "r") as f:
        configs = json.load(f)
    
    model_configs = configs["model_configs"]
    print(f"Model configs: {model_configs}")
    
    # Find checkpoint file
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.ckpt')]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found in {model_dir}")
    
    checkpoint_path = os.path.join(model_dir, checkpoint_files[0])
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = Query2Label_pl.load_from_checkpoint(
        checkpoint_path,
        strict=False
    )
    model.eval()
    model = model.to(device)
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())} parameters")
    print("Using model with internal projection layers")
    
    num_blm_tokens = int(model_configs.get('num_blm_tokens', 32))
    
    return model, num_blm_tokens, configs


def run_inference(model, num_blm_tokens, data, configs, device):
    """Run inference on data and return predictions array."""
    print("Creating dataset and dataloader...")
    
    # Use all data for inference
    all_indices = list(range(len(data['seq_2_terms'])))
    test_dataset = EmbeddingsTestDataset(data, oversample_indices=all_indices)
    
    batch_size = configs['training_configs'].get('batch_size', 64)
    num_workers = configs['training_configs'].get('num_workers', 0)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=simple_collate
    )
    
    print("Using raw features loader (model has internal projections)")
    test_loader = PrefetchLoaderWithRawFeatures(
        test_loader, device,
        num_blm_tokens=num_blm_tokens
    )
    
    print(f"Dataset created with {len(test_dataset)} samples")
    print(f"Running inference...")
    
    # Store predictions
    predictions_list = []
    true_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            x = batch['go_embed']
            true_labels = batch['label']
            plm_raw = batch['plm_raw']
            blm_raw = batch['blm_raw']
            blm_mask = batch.get('blm_mask', None)
            
            logits = model(x, plm_raw=plm_raw, blm_raw=blm_raw, blm_mask=blm_mask)
            probs = torch.sigmoid(logits)
            
            # Move to CPU and store as numpy arrays
            probs_cpu = probs.cpu().numpy()
            labels_cpu = true_labels.cpu().numpy()
            
            predictions_list.append(probs_cpu)
            true_list.append(labels_cpu)
            
            # Free memory
            del logits, probs, probs_cpu, labels_cpu, x, plm_raw, blm_raw
    
    # Stack all predictions
    predictions_array = np.vstack(predictions_list)
    true_array = np.vstack(true_list)
    
    print(f"Inference complete! Predictions shape: {predictions_array.shape}")
    
    return predictions_array, true_array, all_indices


def create_predictions_dataframe(predictions_array, entry_ids, terms_lists, threshold=0.5, top_k=None):
    """
    Create predictions DataFrame from arrays.
    
    Args:
        predictions_array: (N, M) array of prediction scores
        entry_ids: List of N entry IDs
        terms_lists: List of N term lists (each with M terms)
        threshold: Only include predictions with score >= threshold
        top_k: If not None, only keep top-k predictions per entry (after threshold filtering)
    
    Returns:
        predictions_df: DataFrame with ['EntryID', 'term', 'score']
    """
    pred_records = []
    
    for i in range(len(entry_ids)):
        entry_id = entry_ids[i]
        terms = terms_lists[i]
        scores = predictions_array[i]
        
        # Filter by threshold and sort by score descending
        above_threshold = scores >= threshold
        filtered_indices = np.where(above_threshold)[0]
        
        # Sort filtered indices by score descending
        if len(filtered_indices) > 0:
            sorted_filtered_indices = filtered_indices[np.argsort(scores[filtered_indices])[::-1]]
            
            # Apply top-k filtering if specified
            if top_k is not None:
                sorted_filtered_indices = sorted_filtered_indices[:top_k]
            
            # Add predictions above threshold
            for idx in sorted_filtered_indices:
                pred_records.append({
                    'EntryID': entry_id,
                    'term': terms[idx],
                    'score': float(scores[idx])
                })
    
    predictions_df = pd.DataFrame(pred_records)
    
    return predictions_df

def main():
    """Main testing loop to generate predictions."""
    # Configuration``
    model_dir = "/mnt/d/ML/Kaggle/CAFA6-new/checkpoints/shared_classifier_F/"
    output_dir = "/mnt/d/ML/Kaggle/CAFA6-new/predictions_output/"
    
    # Probability threshold - only predictions above this will be included
    threshold = 0.05
    
    # Top-k filtering - only keep top k predictions per entry (None = keep all)
    top_k = None
    
    # Data paths
    data_paths =  {
        "seq_2_terms_df":       "/mnt/d/ML/Kaggle/CAFA6-new/data_packet_test/seq_2_terms.parquet",
        "plm_features_path":    "/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/plm_features.npy",
        "prot_2_pmid_path":     "/mnt/d/ML/Kaggle/CAFA6-new/data_packet_test/prot_2_pmid.npy",
        "pmid_2_embed_path":    "/mnt/d/ML/Kaggle/CAFA6-new/data_packet_test/pmid_2_embed.npy",
        "go_embeds_paths":      "/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/go_embeddings.pkl",
        "go_term_to_aspect_path": "/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/go_term_to_aspect.npy",
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, num_blm_tokens, configs = load_model_and_tokenizer(model_dir, device)
    
    # Prepare data
    print("\nPreparing data...")
    max_terms = configs['model_configs'].get('max_terms', 64)
    aspect = configs['training_configs'].get('aspect', 'P')
    
    data = prepare_data_test(data_paths, max_terms=max_terms, aspect=aspect)
    print(f"Data prepared: {len(data['seq_2_terms'])} sequences")
    
    # Run inference
    print("\nRunning inference...")
    predictions_array, true_array, all_indices = run_inference(
        model, num_blm_tokens, data, configs, device
    )
    
    # Extract entry IDs and terms
    entry_ids_list = []
    terms_list = []
    
    for idx in all_indices:
        row = data['seq_2_terms'].iloc[idx]
        entry_ids_list.append(row['qseqid'])
        terms_list.append(row['terms_predicted'])
    
    print(f"\nCollected {len(entry_ids_list)} entries")
    
    # Generate predictions with threshold filtering
    print(f"\n{'='*60}")
    if top_k is not None:
        print(f"Generating predictions with threshold >= {threshold} and top-{top_k} per entry")
    else:
        print(f"Generating predictions with threshold >= {threshold}")
    print(f"{'='*60}")
    
    pred_df = create_predictions_dataframe(
        predictions_array, entry_ids_list, terms_list, threshold=threshold, top_k=top_k
    )
    
    print(f"Total predictions: {len(pred_df)}")
    print(f"Unique proteins: {pred_df['EntryID'].nunique()}")
    if pred_df['EntryID'].nunique() > 0:
        print(f"Avg predictions per protein: {len(pred_df) / pred_df['EntryID'].nunique():.2f}")
    print(f"Score range: [{pred_df['score'].min():.4f}, {pred_df['score'].max():.4f}]")
    
    # Save predictions to file
    if top_k is not None:
        output_file = os.path.join(output_dir, f"predictions_{aspect}_threshold{threshold}_top{top_k}.tsv")
    else:
        output_file = os.path.join(output_dir, f"predictions_{aspect}_threshold{threshold}.tsv")
    pred_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
