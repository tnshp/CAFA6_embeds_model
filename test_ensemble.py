import torch
import numpy as np
import pandas as pd
import os
import json
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from Dataset.Utils import prepare_data_range, read_fasta
from Utils.tokenizer import EmbedTokenizer
from Dataset.EmbeddingsDataset import TokenizedEmbeddingsDataset, collate_tokenize
from Model.Query2Label_pl import Query2Label_pl


def load_ensemble_model(checkpoint_path, tokenizer_path, configs_path, device='cuda'):
    """
    Load a single ensemble member from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        tokenizer_path: Path to tokenizer_state_dict.pt
        configs_path: Path to configs.json
        device: Device to load model on
        
    Returns:
        model: Loaded Query2Label_pl model
        tokenizer: Loaded EmbedTokenizer
        top_terms: List of GO terms this model predicts
    """
    # Load configs
    with open(configs_path, 'r') as f:
        configs = json.load(f)
    
    # Load model from checkpoint
    model = Query2Label_pl.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    
    # Load tokenizer
    # Get the actual embedding dimension from the saved tokenizer
    tokenizer_state = torch.load(tokenizer_path, map_location=device)
    
    # Reconstruct tokenizer with correct dimensions
    # The tokenizer maps from D (input embed dim) to d (token_dim)
    # We need to figure out D from the saved state
    if 'fc_in.weight' in tokenizer_state:
        D = tokenizer_state['fc_in.weight'].shape[1]
        d = tokenizer_state['fc_in.weight'].shape[0]
        N = tokenizer_state['tokens'].shape[0]
    elif 'P_buffer' in tokenizer_state:
        # P_buffer has shape (N, d, D)
        N = tokenizer_state['P_buffer'].shape[0]
        d = tokenizer_state['P_buffer'].shape[1]
        D = tokenizer_state['P_buffer'].shape[2]
    else:
        # Fallback to config values
        D = configs.get('data_paths', {}).get('embedding_dim', 1280)
        d = configs['model_configs'].get('token_dim', 256)
        N = configs['model_configs'].get('num_tokens', 32)
    
    print(f"  Tokenizer dimensions: D={D} (input), d={d} (token), N={N} (num tokens)")
    
    tokenizer = EmbedTokenizer(D=D, d=d, N=N)
    tokenizer.load_state_dict(tokenizer_state)
    tokenizer.eval()
    tokenizer = tokenizer.to(device)
    
    # Load top_terms
    top_terms_path = os.path.join(os.path.dirname(checkpoint_path), 'top_terms.npy')
    top_terms = np.load(top_terms_path, allow_pickle=True)
    
    return model, tokenizer, top_terms, configs


def predict_ensemble(ensemble_dir, test_embeds, test_ids, device='cuda', batch_size=64):
    """
    Make predictions using all models in an ensemble.
    
    Args:
        ensemble_dir: Path to ensemble directory (e.g., checkpoints/ensemble_first_iter)
        test_embeds: Test embeddings array (N, embed_dim)
        test_ids: Test IDs array (N,)
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        predictions_dict: Dictionary mapping GO term -> predictions array
        all_predictions: Combined predictions array
        all_terms: Combined list of all GO terms
    """
    ensemble_dir = Path(ensemble_dir)
    
    # Find all subdirectories with checkpoints
    model_dirs = sorted([d for d in ensemble_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(model_dirs)} ensemble members in {ensemble_dir}")
    
    # Dictionary to store predictions for each term
    predictions_dict = {}
    all_models_info = []
    
    for model_dir in model_dirs:
        print(f"\nProcessing {model_dir.name}...")
        
        # Find best checkpoint (highest val_fmax_macro)
        ckpt_files = list(model_dir.glob('*.ckpt'))
        if not ckpt_files:
            print(f"  No checkpoint files found in {model_dir}, skipping...")
            continue
        
        # Parse val_fmax_macro from filename and select best
        best_ckpt = None
        best_score = -1
        for ckpt in ckpt_files:
            try:
                # Format: epoch=XX-val_fmax_macro=0.XXXX.ckpt
                score_str = ckpt.stem.split('val_fmax_macro=')[1]
                score = float(score_str)
                if score > best_score:
                    best_score = score
                    best_ckpt = ckpt
            except:
                continue
        
        if best_ckpt is None:
            print(f"  Could not parse checkpoint scores, using first checkpoint")
            best_ckpt = ckpt_files[0]
        else:
            print(f"  Using checkpoint: {best_ckpt.name} (val_fmax_macro={best_score:.4f})")
        
        # Load model, tokenizer, and top_terms
        tokenizer_path = model_dir / 'tokenizer_state_dict.pt'
        configs_path = model_dir / 'configs.json'
        
        # Handle .tmp config files
        if not configs_path.exists():
            configs_path = model_dir / 'configs.json.tmp'
        
        if not tokenizer_path.exists():
            print(f"  Tokenizer not found in {model_dir}, skipping...")
            continue
        
        if not configs_path.exists():
            print(f"  Configs not found in {model_dir}, skipping...")
            continue
        
        model, tokenizer, top_terms, configs = load_ensemble_model(
            str(best_ckpt), str(tokenizer_path), str(configs_path), device
        )
        
        print(f"  Loaded model for {len(top_terms)} GO terms")
        print(f"  Term range: {top_terms[0]} to {top_terms[-1]}")
        
        # Create dataset for inference
        # We need to create a dummy data dict with the test embeddings
        data = {
            'entries': list(test_ids),
            'embeds': test_embeds,
            'labels': np.zeros((len(test_ids), len(top_terms))),  # Dummy labels
            'top_terms': list(top_terms),
            'num_classes': len(top_terms)
        }
        
        test_dataset = TokenizedEmbeddingsDataset(data, oversample_indices=list(range(len(test_ids))))
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False, 
            collate_fn=lambda b: collate_tokenize(b, tokenizer, device)
        )
        
        # Make predictions
        all_logits = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['tokens']
                logits = model(x)
                all_logits.append(logits.cpu())
        
        # Concatenate and apply sigmoid to get probabilities
        all_logits = torch.cat(all_logits, dim=0)
        all_probs = torch.sigmoid(all_logits).numpy()
        
        print(f"  Predictions shape: {all_probs.shape}")
        
        # Store predictions for each term
        for i, term in enumerate(top_terms):
            predictions_dict[term] = all_probs[:, i]
        
        # Store model info
        all_models_info.append({
            'model_dir': str(model_dir),
            'checkpoint': str(best_ckpt),
            'num_terms': len(top_terms),
            'terms': list(top_terms)
        })
    
    # Combine all predictions
    all_terms = sorted(predictions_dict.keys())
    all_predictions = np.stack([predictions_dict[term] for term in all_terms], axis=1)
    
    print(f"\n{'='*60}")
    print(f"Ensemble prediction complete!")
    print(f"Total GO terms: {len(all_terms)}")
    print(f"Total samples: {len(test_ids)}")
    print(f"Predictions shape: {all_predictions.shape}")
    print(f"{'='*60}")
    
    return predictions_dict, all_predictions, all_terms, all_models_info


def save_predictions(predictions, test_ids, terms, output_path, threshold=0.01):
    """
    Save predictions with 3 columns: EntryID, Prediction Term, probability.
    Only entries with probability >= threshold are saved.
    
    Args:
        predictions: Array of shape (N, num_terms)
        test_ids: Array of test IDs (N,)
        terms: List of GO terms
        output_path: Path to save predictions
        threshold: Minimum probability threshold (default: 0.01)
    """
    rows = []
    for i, protein_id in tqdm(enumerate(test_ids)):
        for j, term in enumerate(terms):
            probability = predictions[i, j]
            if probability >= threshold:  # Filter by threshold
                rows.append({
                    'EntryID': protein_id,
                    'Prediction Term': term,
                    'probability': probability
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"Saved predictions to {output_path}")
    print(f"Total predictions: {len(df)} (threshold >= {threshold})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using ensemble models")
    parser.add_argument('--ensemble_dir', type=str, 
                        default='./checkpoints/ensemble_first_iter',
                        help='Path to ensemble directory')
    parser.add_argument('--test_embeds', type=str,
                        default='/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm3_embeds/test_embeds.npy',
                        help='Path to test embeddings')
    parser.add_argument('--test_ids', type=str,
                        default='/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm3_embeds/test_ids.npy',
                        help='Path to test IDs')
    parser.add_argument('--output', type=str,
                        default='./ensemble_predictions.tsv',
                        help='Output path for predictions')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum probability threshold for saving predictions (default: 0.01)')
    
    args = parser.parse_args()
    
    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\nLoading test embeddings from {args.test_embeds}...")
    test_embeds = np.load(args.test_embeds, allow_pickle=True)
    print(f"Test embeddings shape: {test_embeds.shape}")
    
    print(f"Loading test IDs from {args.test_ids}...")
    test_ids = np.load(args.test_ids, allow_pickle=True)
    print(f"Test IDs shape: {test_ids.shape}")
    
    # Make predictions
    predictions_dict, all_predictions, all_terms, models_info = predict_ensemble(
        args.ensemble_dir,
        test_embeds,
        test_ids,
        device=device,
        batch_size=args.batch_size
    )
    
    # Save predictions
    save_predictions(all_predictions, test_ids, all_terms, args.output, threshold=args.threshold)
    
    # Save model info
    info_path = args.output.replace('.tsv', '_models_info.json')
    with open(info_path, 'w') as f:
        json.dump(models_info, f, indent=2)
    print(f"Saved model info to {info_path}")
