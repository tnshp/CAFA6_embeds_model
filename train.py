
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import warnings
import obonet
import os
import json
import argparse

warnings.filterwarnings('ignore')

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from Dataset.Utils import load_data
from Dataset.EmbeddingsDataset import EmbeddingsDataset, simple_collate, PrefetchLoaderWithRawFeatures
from Model.Query2Label_pl import Query2Label_pl

def train_from_configs(configs, run_name=None, resume_from_checkpoint=None, 
                       train_terms_df=None, train_embeds=None, train_ids=None,
                       go_graph=None, ia_map=None, train_seq=None):
        """Train model from configs with optional checkpoint resumption.
        
        Args:
            configs: dict with data_paths, model_configs, training_configs
            run_name: optional run name for logging/checkpointing
            resume_from_checkpoint: optional path to checkpoint .ckpt file to resume training from
            train_terms_df: pre-loaded training terms dataframe (optional, will load if None)
            train_embeds: pre-loaded training embeddings (optional, will load if None)
            train_ids: pre-loaded training IDs (optional, will load if None)
            go_graph: pre-loaded GO graph (optional, will load if None)
            ia_map: pre-loaded information accretion map (optional, will load if None)
            train_seq: pre-loaded training sequences (optional, will load if None)
        """


        model_configs = configs.get('model_configs', {})
        training_configs = configs.get('training_configs', {})

        print(f"Preparing data with aspect {training_configs.get('aspect', None)} max terms {model_configs.get('max_terms', None)}...")
        data = load_data(configs.get('data_paths', {}), max_terms = model_configs.get('max_terms', None), aspect=training_configs.get('aspect', None))
        print("Data preparation complete.")
        
        # Load IA (Information Accretion) scores for weighted F1 computation
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

        
        # Load predefined train/val split from mmseq clustering
        train_ids_path = configs.get('data_paths', {}).get('train_ids_path', None)
        val_ids_path = configs.get('data_paths', {}).get('val_ids_path', None)
        
        all_idx = np.arange(len(data['seq_2_terms']))
        
        if train_ids_path and val_ids_path:
            # Use predefined split from mmseq clustering
            print(f"Loading predefined train/val split from:")
            print(f"  Train IDs: {train_ids_path}")
            print(f"  Val IDs: {val_ids_path}")
            
            # Load train and val IDs
            train_seq_ids = pd.read_csv(train_ids_path, header=None)
            val_seq_ids = pd.read_csv(val_ids_path, header=None)
            
            # Extract the middle part of the ID (between | |)
            train_ids_set = set(train_seq_ids[0].str.split('|').str[1].tolist())
            val_ids_set = set(val_seq_ids[0].str.split('|').str[1].tolist())
            
            print(f"Loaded {len(train_ids_set)} train IDs and {len(val_ids_set)} val IDs")
            
            # Map sequence IDs to indices in data['seq_2_terms']
            seq_ids = data['seq_2_terms']['qseqid'].values
            train_idx = [i for i, seq_id in enumerate(seq_ids) if seq_id in train_ids_set]
            val_idx = [i for i, seq_id in enumerate(seq_ids) if seq_id in val_ids_set]
            
            print(f"Mapped to {len(train_idx)} train indices and {len(val_idx)} val indices")
            
            # For consistency with downstream code
            sampled_idx = list(train_idx)
            unique_train_idx = np.array(train_idx)
            
        else:
            # No sampling_instances provided: split indices randomly into train/val according to val_fraction
            rng = np.random.RandomState(seed) if seed is not None else np.random
            perm = rng.permutation(all_idx)
            n_val = int(np.round(len(all_idx) * val_fraction))
            val_idx = perm[:n_val].tolist()
            train_idx = perm[n_val:].tolist()

            # For consistency with downstream code, set sampled_idx to train indices (no repetitions)
            sampled_idx = list(train_idx)
            unique_train_idx = np.array(train_idx)

            print(f"No sampling_instances set: randomly split {len(all_idx)} indices into {len(train_idx)} train and {len(val_idx)} val (val_fraction={val_fraction}).")

        # Get dimensions for PLM and BLM embeddings
        key = next(iter(data['plm_embeds']))
        plm_embedding_dim = int(np.asarray(data['plm_embeds'][key]).shape[0])
        blm_key = next(iter(data['pmid_2_embed']))
        blm_embedding_dim = int(np.asarray(data['pmid_2_embed'][blm_key]).shape[0])
        
        print(f"PLM embedding dimension: {plm_embedding_dim}")
        print(f"BLM embedding dimension: {blm_embedding_dim}")

        # Model configuration
        token_d = int(model_configs.get('token_dim', 512))
        num_plm_tokens = int(model_configs.get('num_plm_tokens', 32))
        num_blm_tokens = int(model_configs.get('num_blm_tokens', 32))
        
        print("Using raw features - projection layers will be part of the model")

        # datasets: training uses all unique sampled indices (with repetitions), val uses remaining indices
        train_dataset = EmbeddingsDataset(data, oversample_indices=sampled_idx)
        val_dataset   = EmbeddingsDataset(data, oversample_indices=list(val_idx))

        # dataloaders
        batch_size  = int(training_configs.get('batch_size', 32))
        num_workers = int(training_configs.get('num_workers', 0))

        # pin_memory=True for faster host-to-device transfer when num_workers > 0
        pin_memory = True if num_workers > 0 else False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=simple_collate)
        train_loader = PrefetchLoaderWithRawFeatures(train_loader, device, num_blm_tokens=num_blm_tokens)

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory, collate_fn=simple_collate)
        val_loader = PrefetchLoaderWithRawFeatures(val_loader, device, num_blm_tokens=num_blm_tokens)

        # instantiate model
        num_classes = model_configs['max_terms']

        model = Query2Label_pl(num_classes       =num_classes, 
                               in_dim            =token_d,
                               plm_dim           =plm_embedding_dim,
                               blm_dim           =blm_embedding_dim,
                               num_plm_tokens    =num_plm_tokens,
                               nheads            =model_configs.get('nheads', 8),
                               num_encoder_layers=model_configs.get('num_encoder_layers', 1),
                               num_decoder_layers=model_configs.get('num_decoder_layers', 2),
                               dim_feedforward   =model_configs.get('dim_feedforward', 2048),
                               dropout           =model_configs.get('dropout', 0.1),
                               lr                =training_configs.get('lr', 1e-4),
                               weight_decay      =training_configs.get('weight_decay', 1e-5),
                               # loss selection
                               loss_function=training_configs.get('loss_function', 'BCE'),
                               # pass asymmetric loss params from configs
                               gamma_neg=float(training_configs.get('gamma_neg', 4.0)),
                               gamma_pos=float(training_configs.get('gamma_pos', 0.0)),
                               clip     =float(training_configs.get('clip', 0.05)),
                               loss_eps =float(training_configs.get('loss_eps', 1e-8)),
                               disable_torch_grad_focal_loss=bool(training_configs.get('disable_torch_grad_focal_loss', True)),
                               # IA weighting for F1 computation and WBCE loss
                               ia_dict=ia_dict,
                               # Epsilon for WBCE loss
                               epsilon=float(training_configs.get('epsilon', 0.5)),
                               # Decoder: use cross-attention only
                               decoder_cross_attention_only=bool(model_configs.get('decoder_cross_attention_only', False)))

        # compute total steps for scheduler and set on model.hparams
        max_epochs = int(training_configs.get('max_epochs', 10))
        total_steps = int(np.ceil(len(train_dataset) / batch_size)) * max_epochs
        warmup_steps = int(training_configs.get('warmup_steps', max(1, int(0.03 * total_steps))))
        # attach to hparams so configure_optimizers can use them
        try:
            if isinstance(model.hparams, dict):
                model.hparams['total_steps'] = total_steps
                model.hparams['warmup_steps'] = warmup_steps
            else:
                setattr(model.hparams, 'total_steps', total_steps)
                setattr(model.hparams, 'warmup_steps', warmup_steps)
        except Exception:
            pass

        # logging and callbacks
        log_dir = training_configs.get('log_dir', './logs')
        # when resuming, append 'resume' to run_name to differentiate logs
        if resume_from_checkpoint:
            resume_run_name = (run_name or training_configs.get('run_name', 'query2label')) + '_resume'
        else:
            resume_run_name = run_name or training_configs.get('run_name', 'query2label')
        logger = TensorBoardLogger(save_dir=log_dir, name=resume_run_name)

        # Checkpointing: save top-k models by validation macro F1 (higher is better)
        top_k = int(training_configs.get('save_top_k', training_configs.get('checkpoint_top_k', 3)))
        # allow explicit checkpoint directory in configs; otherwise default to logger dir + /checkpoints
        configured_ckpt_dir = training_configs.get('checkpoint_dir', None)
        if configured_ckpt_dir:
            checkpoint_dir = configured_ckpt_dir
        else:
            checkpoint_dir = os.path.join(logger.log_dir, 'checkpoints')

        
        # Create run-specific subdirectory inside checkpoint_dir
        run_name_for_ckpt = run_name or training_configs.get('run_name', 'default_run')
        checkpoint_dir = os.path.join(checkpoint_dir, run_name_for_ckpt)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save a copy of the configs used for this run into the checkpoint directory
        try:
            configs_path = os.path.join(checkpoint_dir, 'configs.json')
            with open(configs_path, 'w') as cf:
                json.dump(configs, cf, indent=2)
            print(f"Saved run configs to: {configs_path}")
            #saving the top terms to checkpoint directory later
            np.save(os.path.join(checkpoint_dir, 'top_terms.npy'), np.array(data['top_terms']))
            print(f"Saved top terms to: {os.path.join(checkpoint_dir, 'top_terms.npy')}")
            
        except Exception as e:
            print(f"Warning: failed to save configs to checkpoint dir: {e}")

        # dirpath ensures checkpoints are saved to the requested folder (not the default lightning logs)
        checkpoint_cb = ModelCheckpoint(dirpath=checkpoint_dir,
                        monitor='val_f1_macro_go', mode='max', save_top_k=top_k,
                        save_last=False,
                        filename='{epoch:02d}-{val_f1_macro_go:.4f}')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # Early stopping: monitor macro F1 (higher is better)
        early_stop = EarlyStopping(monitor='val_f1_macro_go', patience=int(training_configs.get('patience', 5)), mode='max')

        # Gradient accumulation
        accumulate_grad_batches = int(training_configs.get('accumulate_grad_batches', 1))

        trainer = pl.Trainer(max_epochs=max_epochs,
                             accelerator='auto',
                             devices=training_configs.get('devices', None),
                             logger=logger,
                             callbacks=[checkpoint_cb, lr_monitor, early_stop],
                             accumulate_grad_batches=accumulate_grad_batches)

        # train (with optional resume from checkpoint)
        if resume_from_checkpoint is None:
            # try to load from config if not provided via CLI
            resume_from_checkpoint = training_configs.get('resume_checkpoint_path', None)
        
        if resume_from_checkpoint:
            if not os.path.exists(resume_from_checkpoint):
                print(f"Warning: resume checkpoint path does not exist: {resume_from_checkpoint}")
                resume_from_checkpoint = None
            else:
                print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_from_checkpoint)

        # Get best f-max score from checkpoint callback
        best_score = getattr(checkpoint_cb, 'best_model_score', None)
        if best_score is not None:
            best_score = best_score.item() if hasattr(best_score, 'item') else float(best_score)
        else:
            best_score = 0.0
        print(f"Best validation F-max score: {best_score:.4f}")

        return model, trainer, best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Protein GO Classifier with PyTorch Lightning")
    parser.add_argument('--config', type=str, default='./configs_new.json', help='Path to config JSON file')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for logging')
    parser.add_argument('--resume', type=str, default=None, help='Optional path to checkpoint .ckpt file to resume training from')
    args = parser.parse_args()

    configs = json.load(open(args.config))

    
    # run training
    run_name = args.run_name
    resume_from_checkpoint = args.resume
    model, trainer, best_score = train_from_configs(configs, run_name=run_name, resume_from_checkpoint=resume_from_checkpoint)
    print(f"\nTraining completed with best F-max: {best_score:.4f}")


