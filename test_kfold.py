import pandas as pd
import numpy as np
import os
import torch
import json
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
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
    """Prepare test data for inference."""
    
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

    term_lengths = seq_2_terms['terms_predicted'].apply(len)

    # Currently only using sequences with max_terms
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


def load_fold_models(kfold_run_dir, device):
    """Load all fold models from a k-fold run directory.
    
    Args:
        kfold_run_dir: Path to directory containing fold subdirectories (e.g., ./checkpoints/my_run/)
        device: torch device
    
    Returns:
        models: list of loaded models
        configs: config from first fold (assumed same for all)
        num_blm_tokens: number of BLM tokens
    """
    
    kfold_path = Path(kfold_run_dir)
    if not kfold_path.exists():
        raise FileNotFoundError(f"K-fold run directory not found: {kfold_run_dir}")
    
    # Find all fold directories
    fold_dirs = sorted([d for d in kfold_path.iterdir() if d.is_dir() and d.name.startswith('fold')])
    
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No fold directories found in {kfold_run_dir}")
    
    print(f"Found {len(fold_dirs)} fold models in {kfold_run_dir}")
    print(f"Fold directories: {[d.name for d in fold_dirs]}")
    
    models = []
    configs = None
    num_blm_tokens = None
    
    for fold_dir in fold_dirs:
        print(f"\nLoading model from {fold_dir.name}...")
        
        # Load configs
        config_path = fold_dir / "configs.json"
        if not config_path.exists():
            print(f"Warning: configs.json not found in {fold_dir}, skipping")
            continue
            
        with open(config_path, "r") as f:
            fold_configs = json.load(f)
        
        if configs is None:
            configs = fold_configs
            model_configs = configs["model_configs"]
            num_blm_tokens = int(model_configs.get('num_blm_tokens', 32))
        
        # Find checkpoint file
        checkpoint_files = list(fold_dir.glob('*.ckpt'))
        if not checkpoint_files:
            print(f"Warning: No checkpoint file found in {fold_dir}, skipping")
            continue
        
        # Use the best checkpoint (highest score in filename)
        checkpoint_path = sorted(checkpoint_files, key=lambda x: x.name)[-1]
        print(f"Loading checkpoint: {checkpoint_path.name}")
        
        # Load model
        model = Query2Label_pl.load_from_checkpoint(
            str(checkpoint_path),
            strict=False
        )
        model.eval()
        model = model.to(device)
        
        models.append(model)
        print(f"✓ Loaded {fold_dir.name} with {sum(p.numel() for p in model.parameters())} parameters")
    
    if len(models) == 0:
        raise RuntimeError("No models could be loaded!")
    
    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(models)} fold models")
    print(f"{'='*60}")
    
    return models, configs, num_blm_tokens


def run_inference_ensemble(models, num_blm_tokens, data, configs, device):
    """Run inference with ensemble of models and average predictions.
    
    Args:
        models: list of models to ensemble
        num_blm_tokens: number of BLM tokens
        data: data dictionary
        configs: config dictionary
        device: torch device
    
    Returns:
        avg_predictions_array: averaged predictions across all models
        all_indices: indices used for inference
    """
    print("\nCreating dataset and dataloader...")
    
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
    
    test_loader = PrefetchLoaderWithRawFeatures(
        test_loader, device,
        num_blm_tokens=num_blm_tokens
    )
    
    print(f"Dataset created with {len(test_dataset)} samples")
    print(f"Running ensemble inference with {len(models)} models...")
    
    # Store predictions from all models
    all_model_predictions = []
    
    # Run inference with each model
    for model_idx, model in enumerate(models):
        print(f"\nModel {model_idx + 1}/{len(models)} - Running inference...")
        model_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Model {model_idx + 1}/{len(models)}"):
                x = batch['go_embed']
                plm_raw = batch['plm_raw']
                blm_raw = batch['blm_raw']
                blm_mask = batch.get('blm_mask', None)
                
                logits = model(x, plm_raw=plm_raw, blm_raw=blm_raw, blm_mask=blm_mask)
                probs = torch.sigmoid(logits)
                
                # Move to CPU and store
                probs_cpu = probs.cpu().numpy()
                model_predictions.append(probs_cpu)
                
                # Free memory
                del logits, probs, probs_cpu, x, plm_raw, blm_raw
        
        # Stack predictions for this model
        model_predictions_array = np.vstack(model_predictions)
        all_model_predictions.append(model_predictions_array)
        
        print(f"✓ Model {model_idx + 1} predictions shape: {model_predictions_array.shape}")
        
        # Clear GPU cache after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Average predictions across all models
    print(f"\n{'='*60}")
    print(f"Averaging predictions across {len(models)} models...")
    avg_predictions_array = np.mean(all_model_predictions, axis=0)
    
    print(f"Ensemble predictions shape: {avg_predictions_array.shape}")
    print(f"Score range: [{avg_predictions_array.min():.4f}, {avg_predictions_array.max():.4f}]")
    print(f"{'='*60}")
    
    return avg_predictions_array, all_indices


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
    """Main testing loop to generate ensemble predictions from k-fold models."""
    
    parser = argparse.ArgumentParser(description="Generate ensemble predictions from k-fold trained models")
    parser.add_argument('--kfold_dir', type=str, required=True,
                        help='Path to k-fold run directory containing fold subdirectories (e.g., ./checkpoints/my_run/)')
    parser.add_argument('--output_dir', type=str, default='/mnt/d/ML/Kaggle/CAFA6-new/predictions_output/',
                        help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Probability threshold for predictions')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Keep only top-k predictions per entry (None = keep all)')
    parser.add_argument('--data_config', type=str, default=None,
                        help='Optional JSON file with data paths (if None, uses defaults)')
    
    args = parser.parse_args()
    
    # Data paths - can be overridden with config file
    if args.data_config:
        with open(args.data_config, 'r') as f:
            data_paths = json.load(f)['data_paths']
    else:
        data_paths = {
            "seq_2_terms_df":       "/mnt/d/ML/Kaggle/CAFA6-new/data_packet_test/seq_2_terms.parquet",
            "plm_features_path":    "/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/plm_features.npy",
            "prot_2_pmid_path":     "/mnt/d/ML/Kaggle/CAFA6-new/data_packet_test/prot_2_pmid.npy",
            "pmid_2_embed_path":    "/mnt/d/ML/Kaggle/CAFA6-new/data_packet_test/pmid_2_embed.npy",
            "go_embeds_paths":      "/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/go_embeddings.pkl",
            "go_term_to_aspect_path": "/mnt/d/ML/Kaggle/CAFA6-new/data_packet1/go_term_to_aspect.npy",
        }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load all fold models
    print(f"\n{'='*60}")
    print(f"Loading k-fold models from: {args.kfold_dir}")
    print(f"{'='*60}")
    models, configs, num_blm_tokens = load_fold_models(args.kfold_dir, device)
    
    # Prepare data
    print(f"\n{'='*60}")
    print("Preparing test data...")
    print(f"{'='*60}")
    max_terms = configs['model_configs'].get('max_terms', 64)
    aspect = configs['training_configs'].get('aspect', 'F')
    
    data = prepare_data_test(data_paths, max_terms=max_terms, aspect=aspect)
    print(f"Data prepared: {len(data['seq_2_terms'])} sequences")
    
    # Run ensemble inference
    print(f"\n{'='*60}")
    print("Running ensemble inference...")
    print(f"{'='*60}")
    avg_predictions_array, all_indices = run_inference_ensemble(
        models, num_blm_tokens, data, configs, device
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
    if args.top_k is not None:
        print(f"Generating predictions with threshold >= {args.threshold} and top-{args.top_k} per entry")
    else:
        print(f"Generating predictions with threshold >= {args.threshold}")
    print(f"{'='*60}")
    
    pred_df = create_predictions_dataframe(
        avg_predictions_array, entry_ids_list, terms_list, 
        threshold=args.threshold, top_k=args.top_k
    )
    
    print(f"\nPrediction Statistics:")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  Unique proteins: {pred_df['EntryID'].nunique()}")
    if pred_df['EntryID'].nunique() > 0:
        print(f"  Avg predictions per protein: {len(pred_df) / pred_df['EntryID'].nunique():.2f}")
    print(f"  Score range: [{pred_df['score'].min():.4f}, {pred_df['score'].max():.4f}]")
    
    # Determine output filename based on kfold directory name
    kfold_run_name = Path(args.kfold_dir).name
    if args.top_k is not None:
        output_file = os.path.join(args.output_dir, 
                                   f"predictions_{kfold_run_name}_{aspect}_threshold{args.threshold}_top{args.top_k}.tsv")
    else:
        output_file = os.path.join(args.output_dir, 
                                   f"predictions_{kfold_run_name}_{aspect}_threshold{args.threshold}.tsv")
    
    pred_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n{'='*60}")
    print(f"✓ Predictions saved to: {output_file}")
    print(f"{'='*60}")
    
    # Save metadata about the ensemble
    metadata = {
        'kfold_dir': args.kfold_dir,
        'num_models': len(models),
        'aspect': aspect,
        'max_terms': max_terms,
        'threshold': args.threshold,
        'top_k': args.top_k,
        'total_predictions': len(pred_df),
        'unique_proteins': int(pred_df['EntryID'].nunique()),
        'score_range': [float(pred_df['score'].min()), float(pred_df['score'].max())]
    }
    
    metadata_file = output_file.replace('.tsv', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
