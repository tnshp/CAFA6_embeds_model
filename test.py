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
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from Dataset.EmbeddingsDataset import TokenizedEmbeddingsDataset, collate_tokenize
from Utils.tokenizer import EmbedTokenizer
from Model.Query2Label_pl import Query2Label_pl


def prepare_full_test_data(test_ids, test_embeds, top_terms):
    """
    Prepare test data for all samples (not filtered by top_terms).
    
    Args:
        test_ids: Array of test IDs
        test_embeds: Array of test embeddings
        top_terms: List of GO terms the model was trained on
    
    Returns:
        data dict compatible with TokenizedEmbeddingsDataset
    """
    # All test samples are used
    valid_entries = test_ids
    valid_embeds = test_embeds
    
    # Create dummy labels (zeros) since we're just doing inference
    labels_binary = np.zeros((len(valid_entries), len(top_terms)))
    
    return {
        'entries': valid_entries,
        'embeds': valid_embeds,
        'labels': labels_binary,
        'top_terms': list(top_terms),
        'num_classes': len(top_terms)
    }


def load_model(checkpoint_path, tokenizer_path, configs_path, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        tokenizer_path: Path to tokenizer_state_dict.pt
        configs_path: Path to configs.json
        device: Device to load model on
        
    Returns:
        model: Loaded Query2Label_pl model
        tokenizer: Loaded EmbedTokenizer
        top_terms: List of GO terms this model predicts
        configs: Configuration dictionary
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Load configs
    with open(configs_path, 'r') as f:
        configs = json.load(f)
    
    # Load model from checkpoint
    model = Query2Label_pl.load_from_checkpoint(checkpoint_path)
    model.eval()
    model = model.to(device)
    print(f"  Model loaded and moved to {device}")
    
    # Load tokenizer
    tokenizer_state = torch.load(tokenizer_path, map_location=device)
    
    # Reconstruct tokenizer with correct dimensions
    if 'fc_in.weight' in tokenizer_state:
        D = tokenizer_state['fc_in.weight'].shape[1]
        d = tokenizer_state['fc_in.weight'].shape[0]
        N = tokenizer_state['tokens'].shape[0]
    elif 'P_buffer' in tokenizer_state:
        N = tokenizer_state['P_buffer'].shape[0]
        d = tokenizer_state['P_buffer'].shape[1]
        D = tokenizer_state['P_buffer'].shape[2]
    else:
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
    print(f"  Loaded {len(top_terms)} GO terms")
    
    return model, tokenizer, top_terms, configs


def predict(model, tokenizer, test_embeds, test_ids, top_terms, device='cuda', batch_size=64):
    """
    Make predictions using a single model on the full test set.
    
    Args:
        model: Trained model
        tokenizer: Trained tokenizer
        test_embeds: Test embeddings array (N, embed_dim)
        test_ids: Test IDs array (N,)
        top_terms: List of GO terms the model predicts
        device: Device to run on
        batch_size: Batch size for inference
        
    Returns:
        predictions: Array of shape (N, num_terms) with probabilities
    """
    print("\nMaking predictions...")
    print(f"  Test samples: {len(test_ids)}")
    print(f"  Predicting for {len(top_terms)} GO terms")
    
    # Create dataset for inference
    data = prepare_full_test_data(test_ids, test_embeds, top_terms)
    
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
        for batch in tqdm(test_loader, desc="Predicting"):
            x = batch['tokens']
            logits = model(x)
            all_logits.append(logits.cpu())
    
    # Concatenate and apply sigmoid to get probabilities
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = torch.sigmoid(all_logits).numpy()
    
    print(f"  Predictions shape: {all_probs.shape}")
    return all_probs


def calculate_fmax(y_true, y_pred_probs, thresholds=None):
    """
    Calculate F-max score by trying different thresholds.
    
    Args:
        y_true: Binary labels (N, num_classes)
        y_pred_probs: Predicted probabilities (N, num_classes)
        thresholds: List of thresholds to try (default: 0.01 to 0.99 with step 0.01)
    
    Returns:
        fmax: Maximum F1 score
        best_threshold: Threshold that achieved fmax
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    
    fmax = 0.0
    best_threshold = 0.0
    
    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        
        # Calculate F1 for this threshold (macro average)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if f1 > fmax:
            fmax = f1
            best_threshold = threshold
    
    return fmax, best_threshold


def evaluate_predictions(predictions, test_ids, terms, terms_df):
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        predictions: Predicted probabilities (N, num_terms)
        test_ids: Test IDs (N,)
        terms: List of GO terms
        terms_df: DataFrame with columns ['EntryID', 'term'] containing ground truth
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print("Evaluating Predictions")
    print("="*60)
    
    # Create ground truth matrix
    y_true = np.zeros((len(test_ids), len(terms)), dtype=int)
    
    # Map test_ids to indices
    id_to_idx = {protein_id: i for i, protein_id in enumerate(test_ids)}
    term_to_idx = {term: i for i, term in enumerate(terms)}
    
    # Fill ground truth matrix
    print("Building ground truth matrix...")
    matched_annotations = 0
    for _, row in tqdm(terms_df.iterrows(), total=len(terms_df)):
        entry_id = row['EntryID']
        term = row['term']
        
        if entry_id in id_to_idx and term in term_to_idx:
            y_true[id_to_idx[entry_id], term_to_idx[term]] = 1
            matched_annotations += 1
    
    print(f"Ground truth matrix shape: {y_true.shape}")
    print(f"Total annotations in ground truth: {len(terms_df)}")
    print(f"Matched annotations (in model's term set): {matched_annotations}")
    print(f"Total positive labels: {y_true.sum()}")
    print(f"Label density: {y_true.sum() / y_true.size * 100:.4f}%")
    
    # Calculate F-max
    print("\nCalculating F-max...")
    fmax, best_threshold = calculate_fmax(y_true, predictions)
    
    # Calculate metrics at best threshold
    y_pred = (predictions >= best_threshold).astype(int)
    
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    recall_micro = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    f1_micro = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
    
    metrics = {
        'fmax': fmax,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'total_annotations': int(len(terms_df)),
        'matched_annotations': int(matched_annotations),
        'coverage': matched_annotations / len(terms_df) if len(terms_df) > 0 else 0.0
    }
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"F-max:                  {fmax:.4f} (at threshold={best_threshold:.3f})")
    print(f"Accuracy:               {accuracy:.4f}")
    print(f"Coverage:               {metrics['coverage']:.2%} ({matched_annotations}/{len(terms_df)} annotations)")
    print("\nMacro-averaged metrics:")
    print(f"  Precision:            {precision_macro:.4f}")
    print(f"  Recall:               {recall_macro:.4f}")
    print(f"  F1-score:             {f1_macro:.4f}")
    print("\nMicro-averaged metrics:")
    print(f"  Precision:            {precision_micro:.4f}")
    print(f"  Recall:               {recall_micro:.4f}")
    print(f"  F1-score:             {f1_micro:.4f}")
    print("="*60 + "\n")
    
    return metrics


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
    print(f"\nSaving predictions to {output_path}...")
    rows = []
    for i, protein_id in tqdm(enumerate(test_ids), total=len(test_ids), desc="Saving"):
        for j, term in enumerate(terms):
            probability = predictions[i, j]
            if probability >= threshold:
                rows.append({
                    'EntryID': protein_id,
                    'Prediction Term': term,
                    'probability': probability
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(df)} predictions (threshold >= {threshold})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single trained model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Path to checkpoint directory (contains .ckpt, tokenizer, configs). If provided, will use best checkpoint.')
    parser.add_argument('--test_embeds', type=str,
                        default='/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm3_embeds/test_embeds.npy',
                        help='Path to test embeddings')
    parser.add_argument('--test_ids', type=str,
                        default='/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm3_embeds/test_ids.npy',
                        help='Path to test IDs')
    parser.add_argument('--terms_df', type=str, default=None,
                        help='Path to terms TSV file with ground truth labels (columns: EntryID, term)')
    parser.add_argument('--output', type=str,
                        default='./predictions.tsv',
                        help='Output path for predictions')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum probability threshold for saving predictions')
    
    args = parser.parse_args()
    
    # Check device availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}\n")
    
    # Determine checkpoint paths
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        # Find best checkpoint
        ckpt_files = list(checkpoint_dir.glob('*.ckpt'))
        if not ckpt_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Parse val_fmax_macro from filename and select best
        best_ckpt = None
        best_score = -1
        for ckpt in ckpt_files:
            try:
                score_str = ckpt.stem.split('val_fmax_macro=')[1]
                score = float(score_str)
                if score > best_score:
                    best_score = score
                    best_ckpt = ckpt
            except:
                continue
        
        if best_ckpt is None:
            best_ckpt = ckpt_files[0]
            print(f"Could not parse scores, using {best_ckpt.name}")
        else:
            print(f"Using best checkpoint: {best_ckpt.name} (val_fmax_macro={best_score:.4f})")
        
        checkpoint_path = str(best_ckpt)
        tokenizer_path = str(checkpoint_dir / 'tokenizer_state_dict.pt')
        configs_path = str(checkpoint_dir / 'configs.json')
        
        # Check for .tmp config
        if not Path(configs_path).exists():
            configs_path = str(checkpoint_dir / 'configs.json.tmp')
    else:
        checkpoint_path = args.checkpoint
        checkpoint_dir = Path(checkpoint_path).parent
        tokenizer_path = str(checkpoint_dir / 'tokenizer_state_dict.pt')
        configs_path = str(checkpoint_dir / 'configs.json')
        
        if not Path(configs_path).exists():
            configs_path = str(checkpoint_dir / 'configs.json.tmp')
    
    # Load test data
    print(f"Loading test embeddings from {args.test_embeds}...")
    test_embeds = np.load(args.test_embeds, allow_pickle=True)
    print(f"  Shape: {test_embeds.shape}")
    
    print(f"Loading test IDs from {args.test_ids}...")
    test_ids = np.load(args.test_ids, allow_pickle=True)
    print(f"  Shape: {test_ids.shape}\n")
    
    # Load model
    model, tokenizer, top_terms, configs = load_model(
        checkpoint_path, tokenizer_path, configs_path, device
    )
    
    # Make predictions
    predictions = predict(
        model, tokenizer, test_embeds, test_ids, top_terms,
        device=device, batch_size=args.batch_size
    )
    
    # Evaluate if ground truth is provided
    if args.terms_df is not None:
        print(f"\nLoading ground truth from {args.terms_df}...")
        terms_df = pd.read_csv(args.terms_df, sep='\t')
        print(f"  Shape: {terms_df.shape}")
        print(f"  Columns: {list(terms_df.columns)}")
        
        # Evaluate predictions
        metrics = evaluate_predictions(predictions, test_ids, top_terms, terms_df)
        
        # Save metrics to JSON
        metrics_path = args.output.replace('.tsv', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to {metrics_path}\n")
    
    # Save predictions
    save_predictions(predictions, test_ids, top_terms, args.output, threshold=args.threshold)
    
    print("\nTest complete!")
