import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings
import os
import json
import argparse
from tqdm import tqdm

warnings.filterwarnings('ignore')

from Dataset.Utils import prepare_data
from Utils.tokenizer import EmbedTokenizer
from Dataset.EmbeddingsDataset import EmbeddingsDataset, collate_tokenize, PrefetchLoader
from Model.Query2Label_pl import Query2Label_pl


def compute_fmax(predictions_df, ground_truth_df, thresholds=None):
    """
    Compute F-max score across different thresholds with memory-efficient processing.
    
    Args:
        predictions_df: DataFrame with columns ['EntryID', 'term', 'score']
        ground_truth_df: DataFrame with columns ['EntryID', 'term']
        thresholds: List of thresholds to try (default: 0.0 to 1.0 in 101 steps)
    
    Returns:
        dict with fmax, best_threshold, precision, recall
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    
    print("Building ground truth dictionary...")
    # Build ground truth set per EntryID (only once)
    gt_dict = {}
    for entry_id, group in ground_truth_df.groupby('EntryID'):
        gt_dict[entry_id] = set(group['term'].values)
    
    print("Pre-sorting predictions by EntryID...")
    # Sort predictions by EntryID for efficient grouping
    predictions_sorted = predictions_df.sort_values('EntryID')
    
    best_f1 = 0.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    
    print(f"Testing {len(thresholds)} thresholds...")
    for idx, threshold in enumerate(tqdm(thresholds, desc="Computing F-max")):
        tp = 0
        fp = 0
        fn = 0
        
        # Filter predictions by threshold (memory efficient)
        pred_above_threshold = predictions_sorted[predictions_sorted['score'] >= threshold]
        
        # Build prediction dict only for this threshold
        pred_dict = {}
        if len(pred_above_threshold) > 0:
            for entry_id, group in pred_above_threshold.groupby('EntryID'):
                pred_dict[entry_id] = set(group['term'].values)
        
        # Get all entry IDs (use set from ground truth as baseline)
        all_entries = set(gt_dict.keys())
        if len(pred_dict) > 0:
            all_entries = all_entries | set(pred_dict.keys())
        
        # Calculate TP, FP, FN
        for entry_id in all_entries:
            true_terms = gt_dict.get(entry_id, set())
            pred_terms = pred_dict.get(entry_id, set())
            
            tp += len(true_terms & pred_terms)
            fp += len(pred_terms - true_terms)
            fn += len(true_terms - pred_terms)
        
        # Clear pred_dict to free memory
        del pred_dict
        del pred_above_threshold
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    return {
        'fmax': best_f1,
        'threshold': best_threshold,
        'precision': best_precision,
        'recall': best_recall
    }


def test_model(configs, output_path='predictions.tsv', threshold=0.5, calculate_metrics=False):
    """
    Test a trained model and generate predictions.
    
    Args:
        configs: Configuration dictionary with data_paths, model_configs, training_configs
        output_path: Path to save predictions TSV file
        threshold: Threshold for filtering predictions (default: 0.5)
        calculate_metrics: Whether to calculate F-max, precision, recall (requires train_terms_df)
    
    Returns:
        predictions_df: DataFrame with EntryID, term, score columns
    """
    
    print("Loading data...")
    data_paths = configs.get('data_paths', {})
    model_configs = configs.get('model_configs', {})
    training_configs = configs.get('training_configs', {})
    
    # Prepare data
    max_terms = model_configs.get('max_terms', 256)
    data = prepare_data(data_paths, max_terms=max_terms)
    print("Data preparation complete.")
    
    # Load checkpoint path
    checkpoint_path = model_configs.get('model_checkpoint')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Build tokenizer (must match training configuration)
    key = next(iter(data['features_embeds']))
    embedding_dim = int(np.asarray(data['features_embeds'][key]).shape[0])
    token_d = int(model_configs.get('token_dim', 256))
    num_tokens = int(model_configs.get('num_tokens', 100))
    tokenizer = EmbedTokenizer(D=embedding_dim, d=token_d, N=num_tokens)
    
    # Try to load tokenizer state from config path or checkpoint directory
    tokenizer_path = model_configs.get('tokenizer_path')
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from config path: {tokenizer_path}")
        tokenizer.load_state_dict(torch.load(tokenizer_path, map_location='cpu'))
    else:
        # Fall back to checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        tokenizer_state_path = os.path.join(checkpoint_dir, 'tokenizer_state_dict.pt')
        if os.path.exists(tokenizer_state_path):
            print(f"Loading tokenizer from checkpoint directory: {tokenizer_state_path}")
            tokenizer.load_state_dict(torch.load(tokenizer_state_path, map_location='cpu'))
        else:
            print("Warning: Tokenizer state not found, using random initialization")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenizer.to(device)
    
    # Load model from checkpoint
    model = Query2Label_pl.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create dataset (use all data for testing)
    all_indices = list(range(len(data['seq_2_terms'])))
    test_dataset = EmbeddingsDataset(data, oversample_indices=all_indices)
    
    batch_size = int(training_configs.get('batch_size', 32))
    num_workers = int(training_configs.get('num_workers', 0))
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False, 
        collate_fn=lambda b: collate_tokenize(b)
    )
    test_loader = PrefetchLoader(test_loader, device, tokenizer=tokenizer)
    
    print(f"Running inference on {len(test_dataset)} samples...")
    
    # Store predictions - write to temporary file in chunks to avoid memory issues
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv')
    temp_path = temp_file.name
    temp_file.write('EntryID\tterm\tscore\n')  # Write header
    
    batch_count = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x = batch['go_embed']
            f = batch['feature']
            entry_ids = batch['entryID']
            predicted_terms = batch['predicted_terms']
            
            # Forward pass
            logits = model(x, f)
            probs = torch.sigmoid(logits)
            
            # Move to CPU and convert to numpy in one go
            probs_cpu = probs.cpu().numpy()
            
            # Collect predictions for each sample in this batch
            batch_predictions = []
            for sample_idx in range(len(entry_ids)):
                entry_id = entry_ids[sample_idx]
                sample_terms = predicted_terms[sample_idx]
                sample_probs = probs_cpu[sample_idx]
                
                # Create records for each GO term
                for term_idx, go_term in enumerate(sample_terms):
                    score = float(sample_probs[term_idx])
                    batch_predictions.append(f"{entry_id}\t{go_term}\t{score}\n")
            
            # Write batch predictions to temp file
            temp_file.writelines(batch_predictions)
            total_predictions += len(batch_predictions)
            
            # Free memory
            del logits, probs, probs_cpu, batch_predictions
            batch_count += 1
            
            # Periodic cleanup
            if batch_count % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    temp_file.close()
    
    print(f"\nGenerated {total_predictions} predictions")
    
    # Read predictions from temp file and filter by threshold
    print("Loading and filtering predictions...")
    predictions_df = pd.read_csv(temp_path, sep='\t')
    
    print(f"Unique proteins: {predictions_df['EntryID'].nunique()}")
    print(f"Unique GO terms: {predictions_df['term'].nunique()}")
    
    # Filter by threshold and save
    filtered_predictions = predictions_df[predictions_df['score'] >= threshold].copy()
    print(f"Predictions above threshold {threshold}: {len(filtered_predictions)}")
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    filtered_predictions.to_csv(output_path, sep='\t', index=False)
    print(f"Saved predictions to: {output_path}")
    
    # Clean up temp file
    try:
        os.unlink(temp_path)
    except:
        pass
    
    # Keep only filtered predictions in memory for metrics
    # Calculate metrics if requested and train_terms_df is available
    if calculate_metrics:
        train_terms_path = data_paths.get('train_terms_df')
        if train_terms_path and os.path.exists(train_terms_path):
            print("\n" + "="*60)
            print("Computing F-max, Precision, and Recall...")
            print("="*60)
            
            ground_truth_df = pd.read_csv(train_terms_path, sep='\t')
            
            # Filter ground truth to only include test entries
            test_entry_ids = set(predictions_df['EntryID'].unique())
            ground_truth_filtered = ground_truth_df[ground_truth_df['EntryID'].isin(test_entry_ids)]
            
            print(f"Ground truth entries: {len(ground_truth_filtered['EntryID'].unique())}")
            print(f"Ground truth annotations: {len(ground_truth_filtered)}")
            
            # Free memory before computing F-max
            del predictions_df
            
            # Compute F-max with only filtered predictions
            metrics = compute_fmax(filtered_predictions, ground_truth_filtered)
            
            print(f"\nResults:")
            print(f"  F-max:      {metrics['fmax']:.4f}")
            print(f"  Threshold:  {metrics['threshold']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print("="*60)
            
            # Save metrics to JSON
            metrics_path = output_path.replace('.tsv', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved metrics to: {metrics_path}")
        else:
            print("\nWarning: train_terms_df not found, skipping metrics calculation")
    
    return filtered_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Protein GO Classifier")
    parser.add_argument('--config', type=str, default='./configs_test.json', 
                        help='Path to test config JSON file')
    parser.add_argument('--output', type=str, default='./predictions.tsv',
                        help='Path to save predictions TSV file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for filtering predictions (default: 0.5)')
    parser.add_argument('--metrics', action='store_true',
                        help='Calculate F-max, precision, and recall')
    args = parser.parse_args()
    
    # Load configs
    with open(args.config, 'r') as f:
        configs = json.load(f)
    
    # Run testing
    predictions_df = test_model(
        configs, 
        output_path=args.output,
        threshold=args.threshold,
        calculate_metrics=args.metrics
    )
    
    print("\nTesting completed successfully!")
