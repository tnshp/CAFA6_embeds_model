from train import train_from_configs
import argparse 
import json
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Protein GO Classifier with PyTorch Lightning")
    parser.add_argument('--config', type=str, default='./configs.json', help='Path to config JSON file')
    parser.add_argument('--start_range', type=int, default=0, help='Start of class range to train on')
    parser.add_argument('--end_range', type=int, default=256, help='End of class range to train on')
    parser.add_argument('--num_classes', type=int, default=16, help='Number of classes for ensemble')
    parser.add_argument('--run_name', type=str, default=None, help='Optional run name for logging')

    args = parser.parse_args()

    configs = json.load(open(args.config))

    
    # run training
    run_name = args.run_name

    configs['training_configs']['log_dir'] = f"./lightning_logs/ensemble_{run_name}" if run_name is not None else "./lightning_logs/ensemble"
    configs['training_configs']['checkpoint_dir'] = f"./checkpoints/ensemble_{run_name}" if run_name is not None else "./checkpoints/ensemble"

    idx_range = np.arange(args.start_range, args.end_range + 1, args.num_classes)

    print(idx_range)
    for i in range(len(idx_range) - 1):
        start_idx = idx_range[i]
        end_idx = idx_range[i+ 1]
        print(f"Training on class range: {start_idx} to {end_idx}")
        
        # Update model configs for current class range
        configs['model_configs']['k_range'] = [start_idx, end_idx]
        
         # Create a unique run name for each ensemble member
        if run_name is not None:
            current_run_name = f"classes_{start_idx}_{end_idx}"
        else:
            current_run_name = None
        
         # Train the model for the current class range
        configs['model_configs']['k_range'] = [start_idx, end_idx]
        train_from_configs(configs, run_name=current_run_name)
