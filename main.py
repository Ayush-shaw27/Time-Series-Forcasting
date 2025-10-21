import os
import sys
import argparse
import time
import random
from datetime import datetime

# Ensure we can import the supervised PatchTST code when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PatchTST_supervised'))

import numpy as np
import torch

# Use a non-interactive backend for matplotlib (safe for headless/servers)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exp.exp_main import Exp_Main

def set_global_seed(seed: int = 2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleNamespace:
    """A minimal args container compatible with Exp_Main expectations."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def build_args(dataset_cfg, common):
    """Compose an args object with fields mirroring PatchTST_supervised/run_longExp.py.

    dataset_cfg: dict with dataset-specific hyperparameters
    common: dict with common/demo-oriented hyperparameters
    """
    # Determine device usage
    use_gpu = common.get('use_gpu', True) and torch.cuda.is_available()

    # Some safe defaults for fast demo runs
    args = SimpleNamespace(
        # Basic
        random_seed=common.get('random_seed', 2021),
        is_training=1,
        model_id=dataset_cfg['model_id'],
        model='PatchTST',

        # Data loader
        data=dataset_cfg['data_key'],              # 'ETTh1' or 'custom'
        root_path=dataset_cfg['root_path'],        # './dataset/'
        data_path=dataset_cfg['data_path'],        # e.g., 'ETTh1.csv'
        features=dataset_cfg.get('features', 'M'), # 'M' for multivariate
        target=dataset_cfg.get('target', 'OT'),
        freq=dataset_cfg.get('freq', 'h'),
        checkpoints=os.path.join(common['save_dir'], dataset_cfg['name'], 'checkpoints'),

        # Forecasting task
        seq_len=common.get('seq_len', 96),
        label_len=common.get('label_len', 48),
        pred_len=common.get('pred_len', 96),

        # PatchTST-specific
        fc_dropout=dataset_cfg.get('fc_dropout', 0.1),
        head_dropout=dataset_cfg.get('head_dropout', 0.0),
        patch_len=dataset_cfg.get('patch_len', 16),
        stride=dataset_cfg.get('stride', 8),
        padding_patch=dataset_cfg.get('padding_patch', 'end'),
        revin=dataset_cfg.get('revin', 1),
        affine=dataset_cfg.get('affine', 0),
        subtract_last=dataset_cfg.get('subtract_last', 0),
        decomposition=dataset_cfg.get('decomposition', 0),
        kernel_size=dataset_cfg.get('kernel_size', 25),
        individual=dataset_cfg.get('individual', 0),

        # Formers
        embed_type=0,
        enc_in=dataset_cfg['enc_in'],
        dec_in=dataset_cfg['enc_in'],
        c_out=dataset_cfg['enc_in'],
        d_model=dataset_cfg.get('d_model', 64),
        n_heads=dataset_cfg.get('n_heads', 4),
        e_layers=dataset_cfg.get('e_layers', 2),
        d_layers=1,
        d_ff=dataset_cfg.get('d_ff', 128),
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=dataset_cfg.get('dropout', 0.2),
        embed='timeF',
        activation='gelu',
        output_attention=False,
        do_predict=False,

        # Optimization
        num_workers=common.get('num_workers', 0),  # 0 is safer on Windows
        itr=1,
        train_epochs=common.get('train_epochs', 5),
        batch_size=dataset_cfg.get('batch_size', common.get('batch_size', 16)),
        patience=common.get('patience', 3),
        learning_rate=common.get('learning_rate', 1e-4),
        des='Demo',
        loss='mse',
        lradj=dataset_cfg.get('lradj', 'TST'),
        pct_start=dataset_cfg.get('pct_start', 0.2),
        use_amp=False,

        # GPU
        use_gpu=use_gpu,
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        test_flop=False,
    )

    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoints, exist_ok=True)

    # Fix seed
    set_global_seed(args.random_seed)

    return args


def run_train_and_test(args: SimpleNamespace):
    """Train and test using Exp_Main, return the experiment and setting name."""
    Exp = Exp_Main
    # Build setting string same as run_longExp for consistency
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        0,
    )

    exp = Exp(args)
    print(f'>>>>>>> start training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)

    print(f'>>>>>>> testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting)

    # Release cached GPU memory (if any)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return exp, setting


def predict_and_plot(exp: Exp_Main, args: SimpleNamespace, save_base: str, max_batches: int = 1):
    """Run a small prediction pass on test data and save a simple plot.

    Saves:
      - PNG plot comparing ground truth vs prediction on the first channel
      - NPY arrays of preds and trues for the first batch
    """
    # Prepare output dirs
    plot_dir = os.path.join(save_base, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    npy_dir = os.path.join(save_base, 'npy')
    os.makedirs(npy_dir, exist_ok=True)

    # Get test loader
    test_data, test_loader = exp._get_data(flag='test')

    exp.model.eval()
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for bi, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if bi >= max_batches:
                break
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)

            # Forward (PatchTST path)
            outputs = exp.model(batch_x)
            # Select appropriate dimension slice, as in exp_main
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            preds_list.append(pred)
            trues_list.append(true)

    if not preds_list:
        print('No predictions generated for plotting.')
        return

    preds = np.array(preds_list)  # [Batches, B, pred_len, C]
    trues = np.array(trues_list)

    # Use the first batch and first sample for a simple visualization
    first_pred = preds[0][0]  # [pred_len, C]
    first_true = trues[0][0]

    # Optionally inverse-transform for better interpretability if scaler exists
    try:
        inv_pred = test_data.inverse_transform(first_pred)
        inv_true = test_data.inverse_transform(first_true)
        y_pred = inv_pred
        y_true = inv_true
    except Exception:
        # If inverse_transform not available or fails, fall back to normalized values
        y_pred = first_pred
        y_true = first_true

    # Plot the first channel
    ch = 0
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:, ch], label='Ground Truth', linewidth=2)
    plt.plot(y_pred[:, ch], label='Prediction', linewidth=2)
    plt.title(f'{args.model_id} | pred_len={args.pred_len} | channel={ch}')
    plt.xlabel('Horizon step')
    plt.ylabel('Value')
    plt.legend()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(plot_dir, f'{args.model_id}_pred_plot_{ts}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Save arrays for inspection
    np.save(os.path.join(npy_dir, f'{args.model_id}_pred.npy'), y_pred)
    np.save(os.path.join(npy_dir, f'{args.model_id}_true.npy'), y_true)

    print(f'Saved prediction plot to: {plot_path}')


def main():
    parser = argparse.ArgumentParser(description='Quick demo runner for PatchTST on ETTh1, weather, traffic')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs for the demo')
    parser.add_argument('--batch_size', type=int, default=5, help='Default batch size (overridden per dataset if set there)')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='Label length')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction length')
    parser.add_argument('--num_workers', type=int, default=0, help='Dataloader workers (0 is safest on Windows)')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='Where the demo datasets are located')
    parser.add_argument('--save_dir', type=str, default='./demo_outputs', help='Base directory to save checkpoints and plots')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Force GPU if available')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')

    args_cli = parser.parse_args()

    # Validate dataset files exist
    dataset_dir = args_cli.root_path
    expected_files = ['ETTh1.csv', 'weather.csv', 'traffic.csv']
    missing = [f for f in expected_files if not os.path.exists(os.path.join(dataset_dir, f))]
    if missing:
        raise FileNotFoundError(f'Missing dataset files in {dataset_dir}: {missing}')

    # Common demo settings
    common = {
        'random_seed': args_cli.seed,
        'use_gpu': args_cli.use_gpu,
        'save_dir': args_cli.save_dir,
        'seq_len': args_cli.seq_len,
        'label_len': args_cli.label_len,
        'pred_len': args_cli.pred_len,
        'num_workers': args_cli.num_workers,
        'train_epochs': args_cli.epochs,
        'batch_size': args_cli.batch_size,
        'patience': 3,
        'learning_rate': 1e-4,
    }

    # Dataset-specific lightweight configs (optimized for a quick demo)
    datasets = [
        {
            'name': 'ETTh1',
            'model_id': 'ETTh1',
            'data_key': 'ETTh1',
            'root_path': dataset_dir,
            'data_path': 'ETTh1.csv',
            'enc_in': 7,  # number of variables
            'features': 'M',
            'freq': 'h',
            # lightweight model settings
            'd_model': 32,
            'n_heads': 4,
            'e_layers': 2,
            'd_ff': 128,
            'dropout': 0.2,
            'fc_dropout': 0.2,
            'batch_size': 32,
            'patch_len': 16,
            'stride': 8,
        },
        {
            'name': 'weather',
            'model_id': 'weather',
            'data_key': 'custom',
            'root_path': dataset_dir,
            'data_path': 'weather.csv',
            'enc_in': 21,
            'features': 'M',
            'freq': 'h',
            'd_model': 64,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 256,
            'dropout': 0.2,
            'fc_dropout': 0.2,
            'batch_size': 16,
            'patch_len': 16,
            'stride': 8,
        },
        {
            'name': 'traffic',
            'model_id': 'traffic',
            'data_key': 'custom',
            'root_path': dataset_dir,
            'data_path': 'traffic.csv',
            'enc_in': 862,
            'features': 'M',
            'freq': 'h',
            'd_model': 64,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 256,
            'dropout': 0.2,
            'fc_dropout': 0.2,
            'batch_size': 4,  # keep small for memory/time
            'patch_len': 16,
            'stride': 8,
            'lradj': 'TST',
            'pct_start': 0.2,
        },
    ]

    all_runs = []

    for ds in datasets:
        # Special per-dataset save base
        save_base = os.path.join(args_cli.save_dir, ds['name'])
        os.makedirs(save_base, exist_ok=True)

        # Build args and run
        args = build_args(ds, common)
        exp, setting = run_train_and_test(args)

        # Extra: produce a PNG plot and save small numpy outputs under save_base
        predict_and_plot(exp, args, save_base, max_batches=1)

        # Track runs
        all_runs.append((ds['name'], setting, save_base))

    print('\nSummary of demo runs:')
    for name, setting, save_base in all_runs:
        print(f'- {name}: setting={setting}, outputs={save_base}')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Finished. Total runtime: {time.time() - t0:.2f}s')