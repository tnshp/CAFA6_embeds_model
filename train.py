
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

from Dataset.Utils import prepare_data_range, read_fasta
from Utils.tokenizer import EmbedTokenizer
from Dataset.Resample import resample
from Dataset.EmbeddingsDataset import TokenizedEmbeddingsDataset, collate_tokenize
from Model.Query2Label_pl import Query2Label_pl

def train_from_configs(configs, run_name=None, resume_from_checkpoint=None):
        """Train model from configs with optional checkpoint resumption.
        
        Args:
            configs: dict with data_paths, model_configs, training_configs
            run_name: optional run name for logging/checkpointing
            resume_from_checkpoint: optional path to checkpoint .ckpt file to resume training from
        """
        # Set paths and load metadata
        data_paths = configs.get('data_paths', {})
        BASE_PATH = data_paths.get('base_path', "./cafa-6-protein-function-prediction/")
        go_graph = obonet.read_obo(os.path.join(BASE_PATH, 'Train/go-basic.obo'))
        print(f"Gene Ontology graph loaded with {len(go_graph)} nodes and {len(go_graph.edges)} edges.")
        train_terms_df = pd.read_csv(os.path.join(BASE_PATH, 'Train/train_terms.tsv'), sep='\t')
        print(f"Training terms loaded. Shape: {train_terms_df.shape}")
        train_fasta_path = os.path.join(BASE_PATH, 'Train/train_sequences.fasta')
        print(f"Training sequences path set: {train_fasta_path}")
        ia_df = pd.read_csv(os.path.join(BASE_PATH, 'IA.tsv'), sep='\t', header=None, names=['term_id', 'ia_score'])
        ia_map = dict(zip(ia_df['term_id'], ia_df['ia_score']))
        print(f"Information Accretion scores loaded for {len(ia_map)} terms.")

        # read sequences (not used directly here but keep for compatibility)
        train_seq = read_fasta(train_fasta_path)

        # configs
        model_configs = configs.get('model_configs', {})
        training_configs = configs.get('training_configs', {})

        # loading embeddings
        print("Loading training embeddings...")
        embeds_path = data_paths.get('embeds_path', '/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm2_embeds_cafa5/train_embeddings.npy')
        ids_path = data_paths.get('ids_path', '/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm2_embeds_cafa5/train_ids.npy')

        train_embeds = np.load(embeds_path, allow_pickle=True) 
        train_ids = np.load(ids_path, allow_pickle=True)
        print(f"Training embeddings loaded. Num samples: {train_embeds.shape[0]}, dim: {train_embeds.shape[1:]}")

        # preparing data (use numpy-based prepare_data_range to avoid large pandas DataFrames)
        k_range = model_configs.get('k_range', [0, 64])
        data = prepare_data_range(train_terms_df, train_ids, train_embeds, k_range)
        print(f"Prepared data with {len(data['entries'])} entries and {data['num_classes']} classes for top_k range {k_range}.")

        # resample / oversample indices for training
        sampled_idx = resample(data, train_terms_df, strategy=training_configs.get('sampling_strategy', ''), I=training_configs.get('sampling_instances', 100000))
        if not sampled_idx:
            print("Warning: resample returned no indices (empty). Falling back to using all available indices for training.")
            sampled_idx = list(range(len(data['entries'])))

        print(f"Resampled {len(sampled_idx)} indices for training (with repetitions).")

        # use all UNIQUE indices in sampled_idx for training (class-balanced)
        unique_train_idx = np.unique(sampled_idx)
        print(f"Unique training indices: {len(unique_train_idx)}")

        # split remaining indices for validation (no test set)
        all_idx = np.arange(len(data['entries']))
        remaining_idx = np.setdiff1d(all_idx, unique_train_idx)
        val_idx = remaining_idx.tolist()
        print(f"Val indices: {len(val_idx)} (using all remaining indices), Test set removed")

        # build tokenizer
        embedding_dim = int(np.asarray(data['embeds'][0]).shape[0])
        token_d = int(model_configs.get('token_dim', 256))
        num_tokens = int(model_configs.get('num_tokens', 100))
        tokenizer = EmbedTokenizer(D=embedding_dim, d=token_d, N=num_tokens)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            tokenizer = tokenizer.to(device)
        except Exception:
            pass

        # datasets: training uses all unique sampled indices (with repetitions), val uses remaining indices
        train_dataset = TokenizedEmbeddingsDataset(data, oversample_indices=sampled_idx)
        val_dataset = TokenizedEmbeddingsDataset(data, oversample_indices=list(val_idx))

        # dataloaders
        batch_size = int(training_configs.get('batch_size', 32))
        num_workers = int(training_configs.get('num_workers', 0))
        # pin_memory should be False since collate_tokenize moves tensors to device directly
        pin_memory = False

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory, collate_fn=lambda b: collate_tokenize(b, tokenizer, device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=pin_memory, collate_fn=lambda b: collate_tokenize(b, tokenizer, device))
        # No test loader (test set removed)

        # instantiate model
        num_classes = data['num_classes']
        model = Query2Label_pl(num_classes=num_classes, in_dim=token_d,
                               nheads=model_configs.get('nheads', 8),
                               num_encoder_layers=model_configs.get('num_encoder_layers', 1),
                               num_decoder_layers=model_configs.get('num_decoder_layers', 2),
                               dim_feedforward=model_configs.get('dim_feedforward', 2048),
                               dropout=model_configs.get('dropout', 0.1),
                               use_positional_encoding=model_configs.get('use_positional_encoding', True),
                               lr=training_configs.get('lr', 1e-4),
                               weight_decay=training_configs.get('weight_decay', 1e-5))

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
        top_k = int(training_configs.get('checkpoint_top_k', 3))
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
        except Exception as e:
            print(f"Warning: failed to save configs to checkpoint dir: {e}")

        # dirpath ensures checkpoints are saved to the requested folder (not the default lightning logs)
        checkpoint_cb = ModelCheckpoint(dirpath=checkpoint_dir,
                        monitor='val_fmax_macro', mode='max', save_top_k=top_k,
                        save_last=False,
                        filename='{epoch:02d}-{val_fmax_macro:.4f}')
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # Early stopping: monitor F-max (higher is better)
        early_stop = EarlyStopping(monitor='val_fmax_macro', patience=int(training_configs.get('patience', 5)), mode='max')

        trainer = pl.Trainer(max_epochs=max_epochs,
                             accelerator='auto',
                             devices=training_configs.get('devices', None),
                             logger=logger,
                             callbacks=[checkpoint_cb, lr_monitor, early_stop])

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

        # Save tokenizer once (only if a best checkpoint was found)
        try:
            best_ckpt = getattr(checkpoint_cb, 'best_model_path', None)
            if best_ckpt:
                save_dir = os.path.dirname(best_ckpt)
                os.makedirs(save_dir, exist_ok=True)

                # state_dict saved with tensors moved to CPU to avoid device issues on load
                tok_state = tokenizer.state_dict()
                tok_state_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in tok_state.items()}
                tokenizer_state_path = os.path.join(save_dir, 'tokenizer_state_dict.pt')
                torch.save(tok_state_cpu, tokenizer_state_path)

                # also save the full tokenizer object (pickled) for convenience
                tokenizer_obj_path = os.path.join(save_dir, 'tokenizer_object.pt')
                try:
                    torch.save(tokenizer, tokenizer_obj_path)
                except Exception:
                    # fallback: save only the state dict if object pickling fails
                    pass

                print(f"Saved tokenizer state to: {tokenizer_state_path}")
            else:
                print("No best checkpoint found; skipping tokenizer save (tokenizer is unchanged during training).")
        except Exception as e:
            print(f"Warning: failed to save tokenizer: {e}")

        # No separate test set: evaluation can be performed on validation set or with a separate script.

        return model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Protein GO Classifier with PyTorch Lightning")
    parser.add_argument('--config', type=str, default='./configs.json', help='Path to config JSON file')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for logging')
    parser.add_argument('--resume', type=str, default=None, help='Optional path to checkpoint .ckpt file to resume training from')
    args = parser.parse_args()

    configs = json.load(open(args.config))

    
    # run training
    run_name = args.run_name
    resume_from_checkpoint = args.resume
    train_from_configs(configs, run_name=run_name, resume_from_checkpoint=resume_from_checkpoint)



