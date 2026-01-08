import pandas as pd
import numpy as np
import os
import torch
import json
import pickle
import obonet
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from Model.Query2Label_pl import Query2Label_pl
from Utils.tokenizer import EmbedTokenizer
from Dataset.EmbeddingsDataset import collate_tokenize, PrefetchLoader
from Dataset.Utils import pad_dataframe_terms


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

        feature_embed = self.data['features_embeds'][qseqid]
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
            'feature'   : feature_embed,
            'go_embed'  : go_embeds,
            'label'     : label,
            'predicted_terms': valid_terms,
            'true_terms': true_terms
        }


def prepare_data_test(data_paths, max_terms=256, aspect=None):
    
    knn_terms_df = data_paths['knn_terms_df']
    train_terms_df = data_paths.get('train_terms_df', None)
    features_embeds_path = data_paths['features_embeds_path']
    features_ids_path = data_paths['features_ids_path']

    go_embeds_paths = data_paths['go_embeds_paths']

    seq_2_terms = pd.read_parquet(knn_terms_df, engine='fastparquet')

    if train_terms_df is not None:
        train_terms = pd.read_csv(train_terms_df, sep='\t')

    

    go_graph = obonet.read_obo(data_paths['go_obo_path'])
        
    with open(go_embeds_paths, 'rb') as f:
        data = pickle.load(f)
        embeddings_dict = data['embeddings']
        go_ids = data['go_ids']
    
    # Filter to keep only terms from a specific aspect if aspect is provided
    if aspect is not None:
        if aspect == 'P':
            aspect = 'biological_process'
        elif aspect == 'F':
            aspect = 'molecular_function'
        elif aspect == 'C':
            aspect = 'cellular_component'
        else:
            raise ValueError(f"Invalid aspect: {aspect}. Must be one of 'P', 'F', 'C'.")
        
        seq_2_terms['terms_predicted'] = seq_2_terms['terms_predicted'].apply(
            lambda terms: [t for t in terms if go_graph.nodes.get(t, {'namespace': None})['namespace'] == aspect]
        )
        # Remove rows where terms_predicted or terms_true is now empty
        seq_2_terms = seq_2_terms[seq_2_terms['terms_predicted'].apply(len) > 0]


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
           'features_embeds': features_embeds_dict,
           'go_embeds': embeddings_dict,
           'go_graph': go_graph
        }  

    return out


def load_model_and_tokenizer(model_dir, device):
    """Load model and tokenizer from checkpoint directory."""
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
    
    # Build and load tokenizer
    token_d = int(model_configs.get('token_dim', 512))
    num_tokens = int(model_configs.get('num_tokens', 64))
    
    tokenizer = EmbedTokenizer(D=1280, d=token_d, N=num_tokens)
    
    # Load tokenizer state
    tokenizer_path = os.path.join(model_dir, "tokenizer_state_dict.pt")
    tokenizer_state = torch.load(tokenizer_path, map_location='cpu')
    tokenizer.load_state_dict(tokenizer_state)
    
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    print(f"Tokenizer loaded successfully on {device}")
    
    return model, tokenizer, configs


def run_inference(model, tokenizer, data, configs, device):
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
        collate_fn=lambda b: collate_tokenize(b)
    )
    
    # Wrap with PrefetchLoader for GPU tokenization
    test_loader = PrefetchLoader(test_loader, device, tokenizer=tokenizer)
    
    print(f"Dataset created with {len(test_dataset)} samples")
    print(f"Running inference...")
    
    # Store predictions
    predictions_list = []
    true_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Inference")):
            x = batch['go_embed']
            f = batch['feature']
            true_labels = batch['label']
            
            # Forward pass
            logits = model(x, f)
            probs = torch.sigmoid(logits)
            
            # Move to CPU and store as numpy arrays
            probs_cpu = probs.cpu().numpy()
            labels_cpu = true_labels.cpu().numpy()
            
            predictions_list.append(probs_cpu)
            true_list.append(labels_cpu)
            
            # Free memory
            del logits, probs, probs_cpu, labels_cpu, x, f
    
    # Stack all predictions
    predictions_array = np.vstack(predictions_list)
    true_array = np.vstack(true_list)
    
    print(f"Inference complete! Predictions shape: {predictions_array.shape}")
    
    return predictions_array, true_array, all_indices


def create_predictions_dataframe(predictions_array, entry_ids, terms_lists, top_k=None):
    """
    Create predictions DataFrame from arrays.
    
    Args:
        predictions_array: (N, M) array of prediction scores
        entry_ids: List of N entry IDs
        terms_lists: List of N term lists (each with M terms)
        top_k: If provided, only keep top-k predictions per sample
    
    Returns:
        predictions_df: DataFrame with ['EntryID', 'term', 'score']
    """
    pred_records = []
    
    for i in range(len(entry_ids)):
        entry_id = entry_ids[i]
        terms = terms_lists[i]
        scores = predictions_array[i]
        
        # Sort by score descending and get top-k if specified
        if top_k is not None:
            sorted_indices = np.argsort(scores)[::-1][:top_k]
        else:
            sorted_indices = np.arange(len(scores))
        
        # Add predictions (only top-k if specified)
        for idx in sorted_indices:
            pred_records.append({
                'EntryID': entry_id,
                'term': terms[idx],
                'score': float(scores[idx])
            })
    
    predictions_df = pd.DataFrame(pred_records)
    
    return predictions_df


def main():
    """Main testing loop to generate predictions."""
    # Configuration
    model_dir = "/mnt/d/ML/Kaggle/CAFA6-new/downloads/test/"
    output_dir = "/mnt/d/ML/Kaggle/CAFA6-new/predictions_output/"
    
    top_k = 2
    
    # Data paths
    data_paths = {
        "knn_terms_df":         "/mnt/d/ML/Kaggle/CAFA6-new/uniprot/diamond_knn_predictions_test.parquet",
        "go_obo_path":          "/mnt/d/ML/Kaggle/CAFA6/cafa-6-protein-function-prediction/Train/go-basic.obo",
        "features_embeds_path": "/mnt/d/ML/Kaggle/CAFA6-new/Dataset/archive/protein_embeddings.npy",
        "features_ids_path":    "/mnt/d/ML/Kaggle/CAFA6-new/Dataset/archive/protein_ids.npy",
        "go_embeds_paths":      "/mnt/d/ML/Kaggle/CAFA6-new/uniprot/go_embeddings.pkl"
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer, configs = load_model_and_tokenizer(model_dir, device)
    
    # Prepare data
    print("\nPreparing data...")
    max_terms = configs['model_configs'].get('max_terms', 64)
    aspect = configs['training_configs'].get('aspect', 'P')
    
    data = prepare_data_test(data_paths, max_terms=max_terms, aspect=aspect)
    print(f"Data prepared: {len(data['seq_2_terms'])} sequences")
    
    # Run inference
    print("\nRunning inference...")
    predictions_array, true_array, all_indices = run_inference(
        model, tokenizer, data, configs, device
    )
    
    # Extract entry IDs and terms
    entry_ids_list = []
    terms_list = []
    
    for idx in all_indices:
        row = data['seq_2_terms'].iloc[idx]
        entry_ids_list.append(row['qseqid'])
        terms_list.append(row['terms_predicted'])
    
    print(f"\nCollected {len(entry_ids_list)} entries")
    
    # Fixed top-k value
    
    
    # Generate predictions with top-k=4
    print(f"\n{'='*60}")
    print(f"Generating Top-{top_k} predictions")
    print(f"{'='*60}")
    
    pred_df = create_predictions_dataframe(
        predictions_array, entry_ids_list, terms_list, top_k=top_k
    )
    
    print(f"Total predictions: {len(pred_df)}")
    print(f"Unique proteins: {pred_df['EntryID'].nunique()}")
    print(f"Avg predictions per protein: {len(pred_df) / pred_df['EntryID'].nunique():.2f}")
    
    # Save predictions to file
    output_file = os.path.join(output_dir, f"predictions_{aspect}_top{top_k}.tsv")
    pred_df.to_csv(output_file, sep='\t', index=False)
    print(f"\nPredictions saved to: {output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
