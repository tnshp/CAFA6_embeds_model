import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import warnings
import os
import json
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from Dataset.Utils import load_data
from Dataset.EmbeddingsDataset import EmbeddingsDataset, simple_collate, PrefetchLoaderWithRawFeatures
from Model.Query2Label_pl import Query2Label_pl


def read_fold_ids(fold_path):
    """Read protein IDs from a fold file and extract the middle part (between | |)."""
    fold_ids = pd.read_csv(fold_path, header=None)
    # Extract the middle part of the ID (between | |)
    ids_set = set(fold_ids[0].str.split('|').str[1].tolist())
    return ids_set


def train_single_fold(configs, fold_idx, train_folds, val_fold, kfold_dir, data, ia_dict, 
                      plm_embedding_dim, blm_embedding_dim, run_name=None):
    """Train model on specific fold configuration.
    
    Args:
        configs: dict with data_paths, model_configs, training_configs
        fold_idx: index of current fold (for logging/naming)
        train_folds: list of fold indices to use for training
        val_fold: fold index to use for validation
        kfold_dir: directory containing fold files
        data: pre-loaded data dictionary
        ia_dict: pre-loaded IA dictionary
        plm_embedding_dim: PLM embedding dimension
        blm_embedding_dim: BLM embedding dimension
        run_name: optional run name for organizing fold models (overrides config)
    
    Returns:
        best_score: best validation F1 score achieved
    """
    
    model_configs = configs.get('model_configs', {})
    training_configs = configs.get('training_configs', {})

    print(f"\n{'='*80}")
    print(f"Training Fold {fold_idx}: Train on folds {train_folds}, Validate on fold {val_fold}")
    print(f"{'='*80}\n")

    # Load train and validation IDs from fold files
    train_ids_set = set()
    for fold_num in train_folds:
        fold_path = os.path.join(kfold_dir, f"fold_{fold_num}.txt")
        print(f"Loading train fold {fold_num} from: {fold_path}")
        fold_ids = read_fold_ids(fold_path)
        train_ids_set.update(fold_ids)
        print(f"  Added {len(fold_ids)} IDs from fold {fold_num}")
    
    val_fold_path = os.path.join(kfold_dir, f"fold_{val_fold}.txt")
    print(f"Loading validation fold {val_fold} from: {val_fold_path}")
    val_ids_set = read_fold_ids(val_fold_path)
    print(f"  Loaded {len(val_ids_set)} IDs for validation")
    
    print(f"\nTotal train IDs: {len(train_ids_set)}, validation IDs: {len(val_ids_set)}")
    
    # Map sequence IDs to indices in data['seq_2_terms']
    seq_ids = data['seq_2_terms']['qseqid'].values
    train_idx = [i for i, seq_id in enumerate(seq_ids) if seq_id in train_ids_set]
    val_idx = [i for i, seq_id in enumerate(seq_ids) if seq_id in val_ids_set]
    
    print(f"Mapped to {len(train_idx)} train indices and {len(val_idx)} val indices")
    
    # Model configuration
    token_d = int(model_configs.get('token_dim', 512))
    num_plm_tokens = int(model_configs.get('num_plm_tokens', 32))
    num_blm_tokens = int(model_configs.get('num_blm_tokens', 32))
    
    print("Using raw features - projection layers will be part of the model")

    # Create datasets
    train_dataset = EmbeddingsDataset(data, oversample_indices=train_idx)
    val_dataset = EmbeddingsDataset(data, oversample_indices=list(val_idx))

    # Create dataloaders
    batch_size = int(training_configs.get('batch_size', 32))
    num_workers = int(training_configs.get('num_workers', 0))

    pin_memory = True if num_workers > 0 else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=simple_collate)
    train_loader = PrefetchLoaderWithRawFeatures(train_loader, device, num_blm_tokens=num_blm_tokens)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=simple_collate)
    val_loader = PrefetchLoaderWithRawFeatures(val_loader, device, num_blm_tokens=num_blm_tokens)

    # Instantiate model
    num_classes = model_configs['max_terms']

    model = Query2Label_pl(num_classes=num_classes, 
                           in_dim=token_d,
                           plm_dim=plm_embedding_dim,
                           blm_dim=blm_embedding_dim,
                           num_plm_tokens=num_plm_tokens,
                           nheads=model_configs.get('nheads', 8),
                           num_encoder_layers=model_configs.get('num_encoder_layers', 1),
                           num_decoder_layers=model_configs.get('num_decoder_layers', 2),
                           dim_feedforward=model_configs.get('dim_feedforward', 2048),
                           dropout=model_configs.get('dropout', 0.1),
                           lr=training_configs.get('lr', 1e-4),
                           weight_decay=training_configs.get('weight_decay', 1e-5),
                           loss_function=training_configs.get('loss_function', 'BCE'),
                           gamma_neg=float(training_configs.get('gamma_neg', 4.0)),
                           gamma_pos=float(training_configs.get('gamma_pos', 0.0)),
                           clip=float(training_configs.get('clip', 0.05)),
                           loss_eps=float(training_configs.get('loss_eps', 1e-8)),
                           disable_torch_grad_focal_loss=bool(training_configs.get('disable_torch_grad_focal_loss', True)),
                           ia_dict=ia_dict,
                           epsilon=float(training_configs.get('epsilon', 0.5)))

    # Compute total steps for scheduler
    max_epochs = int(training_configs.get('max_epochs', 10))
    total_steps = int(np.ceil(len(train_dataset) / batch_size)) * max_epochs
    warmup_steps = int(training_configs.get('warmup_steps', max(1, int(0.03 * total_steps))))
    
    try:
        if isinstance(model.hparams, dict):
            model.hparams['total_steps'] = total_steps
            model.hparams['warmup_steps'] = warmup_steps
        else:
            setattr(model.hparams, 'total_steps', total_steps)
            setattr(model.hparams, 'warmup_steps', warmup_steps)
    except Exception:
        pass

    # Logging and callbacks
    log_dir = training_configs.get('log_dir', './lightning_logs')
    # Use provided run_name or fall back to config
    base_run_name = run_name or training_configs.get('run_name', 'kfold_cv')
    # Create run_name directory, then fold subdirectory
    run_log_dir = os.path.join(log_dir, base_run_name)
    fold_name = f"fold{fold_idx}"
    logger = TensorBoardLogger(save_dir=run_log_dir, name=fold_name)

    # Checkpointing
    top_k = int(training_configs.get('save_top_k', 3))
    configured_ckpt_dir = training_configs.get('checkpoint_dir', None)
    if configured_ckpt_dir:
        checkpoint_base_dir = configured_ckpt_dir
    else:
        checkpoint_base_dir = './checkpoints'

    # Create run_name directory, then fold subdirectory
    checkpoint_dir = os.path.join(checkpoint_base_dir, base_run_name, fold_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Save configs and top terms
    try:
        configs_path = os.path.join(checkpoint_dir, 'configs.json')
        fold_config = configs.copy()
        fold_config['fold_info'] = {
            'fold_idx': fold_idx,
            'train_folds': train_folds,
            'val_fold': val_fold
        }
        with open(configs_path, 'w') as cf:
            json.dump(fold_config, cf, indent=2)
        print(f"Saved fold configs to: {configs_path}")
        
        np.save(os.path.join(checkpoint_dir, 'top_terms.npy'), np.array(data['top_terms']))
        print(f"Saved top terms to: {os.path.join(checkpoint_dir, 'top_terms.npy')}")
    except Exception as e:
        print(f"Warning: failed to save configs to checkpoint dir: {e}")

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='val_f1_macro_go',
        mode='max',
        save_top_k=top_k,
        save_last=False,
        filename='{epoch:02d}-{val_f1_macro_go:.4f}'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop = EarlyStopping(
        monitor='val_f1_macro_go',
        patience=int(training_configs.get('patience', 5)),
        mode='max'
    )

    # Gradient accumulation
    accumulate_grad_batches = int(training_configs.get('accumulate_grad_batches', 1))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=training_configs.get('devices', None),
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop],
        accumulate_grad_batches=accumulate_grad_batches
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Get best score
    best_score = getattr(checkpoint_cb, 'best_model_score', None)
    if best_score is not None:
        best_score = best_score.item() if hasattr(best_score, 'item') else float(best_score)
    else:
        best_score = 0.0
    
    print(f"\nFold {fold_idx} - Best validation F-max score: {best_score:.4f}")

    return best_score


def train_kfold(configs, n_folds=None, run_name=None):
    """Train models using k-fold cross-validation.
    
    Args:
        configs: dict with data_paths, model_configs, training_configs
        n_folds: number of folds (if None, will auto-detect from kfold_dir)
        run_name: optional run name for organizing fold models (overrides config)
    
    Returns:
        fold_scores: list of best validation scores for each fold
    """
    
    model_configs = configs.get('model_configs', {})
    training_configs = configs.get('training_configs', {})
    
    kfold_dir = configs.get('data_paths', {}).get('kfold_dir', '/mnt/d/ML/Kaggle/CAFA6-new/mmseq/kfold_splits')
    
    # Auto-detect number of folds if not specified
    if n_folds is None:
        fold_files = sorted([f for f in os.listdir(kfold_dir) if f.startswith('fold_') and f.endswith('.txt')])
        n_folds = len(fold_files)
        print(f"Auto-detected {n_folds} folds in {kfold_dir}")
    
    if n_folds < 2:
        raise ValueError(f"Need at least 2 folds for cross-validation, found {n_folds}")
    
    print(f"\nStarting {n_folds}-fold cross-validation")
    print(f"Fold directory: {kfold_dir}")
    print("="*80)
    
    # Load data once for all folds
    print(f"\nLoading data once for all folds...")
    print(f"Aspect: {training_configs.get('aspect', None)}, Max terms: {model_configs.get('max_terms', None)}")
    data = load_data(configs.get('data_paths', {}), max_terms=model_configs.get('max_terms', None), 
                     aspect=training_configs.get('aspect', None))
    print("Data loading complete.")
    
    # Load IA (Information Accretion) scores once
    ia_dict = None
    ia_file_path = configs.get('data_paths', {}).get('ia_file_path', None)
    if ia_file_path and os.path.exists(ia_file_path):
        print(f"Loading IA scores from {ia_file_path}...")
        ia_df = pd.read_csv(ia_file_path, sep='\t', header=None)
        ia_df.columns = ['terms', 'ia']
        ia_dict = dict(zip(ia_df['terms'], ia_df['ia']))
        print(f"Loaded IA scores for {len(ia_dict)} GO terms")
    else:
        print("No IA file specified or file not found, using unweighted F1")
    
    # Get embedding dimensions once
    key = next(iter(data['plm_embeds']))
    plm_embedding_dim = int(np.asarray(data['plm_embeds'][key]).shape[0])
    blm_key = next(iter(data['pmid_2_embed']))
    blm_embedding_dim = int(np.asarray(data['pmid_2_embed'][blm_key]).shape[0])
    
    print(f"PLM embedding dimension: {plm_embedding_dim}")
    print(f"BLM embedding dimension: {blm_embedding_dim}")
    
    # Determine the run name for this k-fold CV run
    base_run_name = run_name or training_configs.get('run_name', 'kfold_cv')
    print(f"Run name: {base_run_name}")
    print(f"Models will be saved in: {training_configs.get('checkpoint_dir', './checkpoints')}/{base_run_name}/fold<N>/")
    print("="*80)
    
    fold_scores = []
    
    # Train one model for each fold
    for val_fold in range(n_folds):
        # Use all other folds for training
        train_folds = [i for i in range(n_folds) if i != val_fold]
        
        best_score = train_single_fold(
            configs=configs,
            fold_idx=val_fold,
            train_folds=train_folds,
            val_fold=val_fold,
            kfold_dir=kfold_dir,
            data=data,
            ia_dict=ia_dict,
            plm_embedding_dim=plm_embedding_dim,
            blm_embedding_dim=blm_embedding_dim,
            run_name=base_run_name
        )
        
        fold_scores.append(best_score)
        
        # Force garbage collection after each fold to free memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("K-Fold Cross-Validation Summary")
    print("="*80)
    for fold_idx, score in enumerate(fold_scores):
        print(f"Fold {fold_idx}: F-max = {score:.4f}")
    print("-"*80)
    print(f"Mean F-max: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Min F-max:  {np.min(fold_scores):.4f}")
    print(f"Max F-max:  {np.max(fold_scores):.4f}")
    print("="*80)
    
    # Save summary
    summary_path = os.path.join(configs.get('training_configs', {}).get('checkpoint_dir', './checkpoints'), 
                                 base_run_name, 'kfold_summary.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    summary = {
        'n_folds': n_folds,
        'fold_scores': [float(s) for s in fold_scores],
        'mean_score': float(np.mean(fold_scores)),
        'std_score': float(np.std(fold_scores)),
        'min_score': float(np.min(fold_scores)),
        'max_score': float(np.max(fold_scores))
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved k-fold summary to: {summary_path}")
    
    return fold_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Protein GO Classifier with K-Fold Cross-Validation")
    parser.add_argument('--config', type=str, default='./configs_kfold.json', 
                        help='Path to config JSON file')
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of folds (auto-detect if not specified)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for organizing fold models (e.g., "experiment1"). Models saved in checkpoint_dir/<run_name>/fold<N>/')
    args = parser.parse_args()

    configs = json.load(open(args.config))
    
    # Run k-fold cross-validation
    fold_scores = train_kfold(configs, n_folds=args.n_folds, run_name=args.run_name)
    print(f"\nK-fold cross-validation completed!")
    print(f"Average performance: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
