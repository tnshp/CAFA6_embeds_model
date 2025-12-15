from train import train_from_configs
import argparse 
import json
import numpy as np
import pandas as pd
import obonet
import os
from Dataset.Utils import read_fasta
import matplotlib.pyplot as plt
import torch
import gc 


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser(description="Train Protein GO Classifier with PyTorch Lightning")
    parser.add_argument('--config', type=str, default='./configs.json', help='Path to config JSON file')
    parser.add_argument('--start_range', type=int, default=0, help='Start of class range to train on')
    parser.add_argument('--end_range', type=int, default=256, help='End of class range to train on')
    parser.add_argument('--num_classes', type=int, default=16, help='Number of classes for ensemble')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for logging')

    args = parser.parse_args()

    configs = json.load(open(args.config))

    # Load all data ONCE before training ensemble
    print("="*60)
    print("Loading shared data for ensemble training...")
    print("="*60)
    
    data_paths = configs.get('data_paths', {})
    BASE_PATH = data_paths.get('base_path', "./cafa-6-protein-function-prediction/")
    
    # Load GO graph
    print("Loading Gene Ontology graph...")
    go_graph = obonet.read_obo(os.path.join(BASE_PATH, 'Train/go-basic.obo'))
    print(f"Gene Ontology graph loaded with {len(go_graph)} nodes and {len(go_graph.edges)} edges.")
    
    # Load training terms
    print("Loading training terms...")
    train_terms_df = pd.read_csv(os.path.join(BASE_PATH, 'Train/train_terms.tsv'), sep='\t')
    print(f"Training terms loaded. Shape: {train_terms_df.shape}")
    
    # Load IA scores
    print("Loading Information Accretion scores...")
    ia_df = pd.read_csv(os.path.join(BASE_PATH, 'IA.tsv'), sep='\t', header=None, names=['term_id', 'ia_score'])
    ia_map = dict(zip(ia_df['term_id'], ia_df['ia_score']))
    print(f"Information Accretion scores loaded for {len(ia_map)} terms.")
    
    # Load sequences
    print("Loading training sequences...")
    train_fasta_path = os.path.join(BASE_PATH, 'Train/train_sequences.fasta')
    train_seq = read_fasta(train_fasta_path)
    print(f"Training sequences loaded.")
    
    # Load embeddings
    print("Loading training embeddings...")
    embeds_path = data_paths.get('embeds_path', '/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm2_embeds_cafa5/train_embeddings.npy')
    ids_path = data_paths.get('ids_path', '/mnt/d/ML/Kaggle/CAFA6-new/Dataset/esm2_embeds_cafa5/train_ids.npy')
    train_embeds = np.load(embeds_path, allow_pickle=True)
    train_ids = np.load(ids_path, allow_pickle=True)
    print(f"Training embeddings loaded. Num samples: {train_embeds.shape[0]}, dim: {train_embeds.shape[1:]}")
    
    print("="*60)
    print("All shared data loaded successfully!")
    print("="*60)
    print()
    
    # run training
    run_name = args.run_name

    # Set base log and checkpoint directories (without subdirectories)
    base_log_dir = f"./lightning_logs/ensemble_{run_name}" if run_name is not None else "./lightning_logs/ensemble"
    base_checkpoint_dir = f"./checkpoints/ensemble_{run_name}" if run_name is not None else "./checkpoints/ensemble"

    idx_range = np.arange(args.start_range, args.end_range + 1, args.num_classes)

    # List to store best f-max scores for each ensemble member
    ensemble_scores = []
    ensemble_model_names = []

    print(idx_range)
    for i in range(len(idx_range) - 1):
        start_idx = int(idx_range[i])  # Convert numpy int64 to Python int
        end_idx = int(idx_range[i + 1])  # Convert numpy int64 to Python int
        print(f"Training on class range: {start_idx} to {end_idx}")
        
        # Create a unique run name for each ensemble member
        current_run_name = f"classes_{start_idx}_{end_idx}"
        
        # Update configs for current class range
        configs['model_configs']['k_range'] = [start_idx, end_idx]
        configs['training_configs']['log_dir'] = base_log_dir
        configs['training_configs']['checkpoint_dir'] = base_checkpoint_dir
        configs['training_configs']['run_name'] = current_run_name
        
        # Save configs for this ensemble member BEFORE training
        # import os
        # checkpoint_dir = os.path.join(base_checkpoint_dir, current_run_name)
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # configs_save_path = os.path.join(checkpoint_dir, 'configs.json')
        
        # print(f"Saving configs to: {configs_save_path}")
        # with open(configs_save_path, 'w') as f:
        #     json.dump(configs, f, indent=2)
        # print(f"Configs saved successfully")
        
        # Train the model for the current class range with pre-loaded data
        model, trainer, best_score = train_from_configs(configs, run_name=current_run_name,
                                                         train_terms_df=train_terms_df,
                                                         train_embeds=train_embeds,
                                                         train_ids=train_ids,
                                                         go_graph=go_graph,
                                                         ia_map=ia_map,
                                                         train_seq=train_seq)
        
        # Store the best score and model name
        ensemble_scores.append(best_score)
        ensemble_model_names.append(current_run_name)
        print(f"Completed {current_run_name} with best F-max: {best_score:.4f}\n")
        
        # CRITICAL: Free up memory after each model training to prevent OOM
        print("Cleaning up memory...")
        # Move model to CPU and delete
        if model is not None:
            try:
                model.cpu()
            except:
                pass
            del model
        
        # Delete trainer
        if trainer is not None:
            del trainer
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        print("Memory cleanup complete.\n")
    
    # After all training is complete, plot and save the ensemble scores
    print("="*60)
    print("Ensemble Training Complete!")
    print("="*60)
    print(f"\nBest F-max scores for each ensemble member:")
    for name, score in zip(ensemble_model_names, ensemble_scores):
        print(f"  {name}: {score:.4f}")
    print(f"\nMean F-max: {np.mean(ensemble_scores):.4f}")
    print(f"Std F-max: {np.std(ensemble_scores):.4f}")
    print(f"Min F-max: {np.min(ensemble_scores):.4f}")
    print(f"Max F-max: {np.max(ensemble_scores):.4f}")
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    x_positions = np.arange(len(ensemble_model_names))
    plt.bar(x_positions, ensemble_scores, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Ensemble Member', fontsize=12)
    plt.ylabel('Best Validation F-max Score', fontsize=12)
    plt.title(f'Ensemble Performance - {run_name if run_name else "default"}', fontsize=14, fontweight='bold')
    plt.xticks(x_positions, ensemble_model_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add mean line
    mean_score = np.mean(ensemble_scores)
    plt.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot to checkpoint directory
    plot_path = os.path.join(base_checkpoint_dir, 'ensemble_fmax_scores.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Also save scores to JSON file
    scores_data = {
        'model_names': ensemble_model_names,
        'fmax_scores': [float(s) for s in ensemble_scores],
        'statistics': {
            'mean': float(np.mean(ensemble_scores)),
            'std': float(np.std(ensemble_scores)),
            'min': float(np.min(ensemble_scores)),
            'max': float(np.max(ensemble_scores))
        }
    }
    scores_json_path = os.path.join(base_checkpoint_dir, 'ensemble_fmax_scores.json')
    with open(scores_json_path, 'w') as f:
        json.dump(scores_data, f, indent=2)
    print(f"Scores data saved to: {scores_json_path}")
    print("="*60)
