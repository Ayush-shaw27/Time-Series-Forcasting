# fixed_runner.py
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def run_inference_csv_fixed(
    checkpoint_path: str,
    dataset_key: str,
    csv_path: str,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    use_gpu: bool = False
):
    """
    Simplified version that loads model and runs inference without data_provider dependency
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Try different possible keys for model state
        model_state = None
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            # If no standard key found, try to use the checkpoint directly
            model_state = checkpoint
        
        if model_state is None:
            raise ValueError("No model weights found in checkpoint file")
        
        # Extract args if available
        args = checkpoint.get('args', None)
        
        # Set device
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Load and preprocess CSV data
        df = pd.read_csv(csv_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in CSV")
        
        # Use numeric columns as features
        data = df[numeric_cols].values.astype(np.float32)
        
        # Simple normalization
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0) + 1e-8
        normalized_data = (data - data_mean) / data_std
        
        # Prepare input sequence
        if len(normalized_data) < seq_len:
            # Pad if sequence is too short
            padding = np.zeros((seq_len - len(normalized_data), len(numeric_cols)))
            input_sequence = np.vstack([padding, normalized_data])
        else:
            # Use the last seq_len points
            input_sequence = normalized_data[-seq_len:]
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)  # [1, seq_len, n_features]
        
        # Try to load and run the model
        try:
            # Import here to avoid circular imports
            from PatchTST_supervised.exp.exp_main import Exp_Main
            
            if args is None:
                # Create minimal args if not in checkpoint
                class SimpleArgs:
                    def __init__(self):
                        self.seq_len = seq_len
                        self.label_len = label_len
                        self.pred_len = pred_len
                        self.enc_in = len(numeric_cols)
                        self.dec_in = len(numeric_cols)
                        self.c_out = len(numeric_cols)
                        self.features = 'M'
                        self.use_gpu = use_gpu
                        self.device = device
                
                args = SimpleArgs()
            
            # Create experiment and load model
            exp = Exp_Main(args)
            
            # Try to load model state, handling potential key mismatches
            try:
                exp.model.load_state_dict(model_state)
            except Exception as load_error:
                # If direct loading fails, try to handle key mismatches
                print(f"Direct load failed: {load_error}, trying to handle key mismatches...")
                
                # Remove 'module.' prefix if present (from DataParallel)
                new_state_dict = {}
                for k, v in model_state.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v  # Remove 'module.' prefix
                    else:
                        new_state_dict[k] = v
                
                try:
                    exp.model.load_state_dict(new_state_dict)
                except Exception as e2:
                    print(f"Loading with key adjustment failed: {e2}")
                    # If still failing, use fallback
                    raise load_error
            
            exp.model.eval()
            exp.model.to(device)
            
            # Create dummy targets for the model forward pass
            batch_y = torch.zeros(1, pred_len, len(numeric_cols)).to(device)
            
            # Run inference
            with torch.no_grad():
                outputs = exp.model(input_tensor, input_tensor, batch_y, batch_y)
                predictions = outputs.detach().cpu().numpy()[0]  # [pred_len, n_features]
            
        except Exception as model_error:
            # Fallback: Use a simple baseline if model loading fails
            print(f"Model loading failed, using baseline: {model_error}")
            predictions = generate_simple_baseline(input_sequence, pred_len, len(numeric_cols))
        
        # Denormalize predictions
        denorm_predictions = predictions * data_std + data_mean
        
        # Create "true" values for comparison (we don't have future values, so use recent past)
        if len(data) > pred_len:
            true_values = data[-pred_len:]
        else:
            # If data is shorter than prediction length, use the available data
            true_values = np.zeros((pred_len, len(numeric_cols)))
            true_values[:len(data)] = data
        
        return true_values, denorm_predictions
        
    except Exception as e:
        raise Exception(f"Inference failed: {str(e)}")

def generate_simple_baseline(input_sequence, pred_len, n_features):
    """
    Generate simple baseline predictions when model fails to load
    """
    # Use the last value repeated for all prediction steps
    last_values = input_sequence[-1]  # Last timestep values
    predictions = np.tile(last_values, (pred_len, 1))
    return predictions

def find_checkpoints_fixed(search_root: str):
    """
    Find checkpoint files in the search root
    """
    checkpoints = []
    if not os.path.exists(search_root):
        return checkpoints
    
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if file.endswith('.pth') or file.endswith('.ckpt') or file.endswith('.pt'):
                full_path = os.path.join(root, file)
                # Create a readable label
                rel_path = os.path.relpath(full_path, search_root)
                checkpoints.append((rel_path, full_path))
    
    return checkpoints