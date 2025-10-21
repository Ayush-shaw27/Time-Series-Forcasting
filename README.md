# # Overview:
We implemented supervised long-term forecasting with the PatchTST architecture on three public datasets (ETTh1, Weather, Traffic). The project upgrades an older repository to work with the latest Python ecosystem (NumPy 2.x, modern pandas/matplotlib, and nightly CPU-only PyTorch for Python 3.13). We reorganized data, modernized requirements, fixed compatibility issues, and provided reproducible training scripts and outputs.

What we did:
1) Data organization
- Placed ETTh1.csv, weather.csv, traffic.csv under dataset/ at the project root.

2) Script updates
- Updated PatchTST_supervised/scripts/PatchTST/*.sh to point to the root dataset/ folder.
- Fixed model_id concatenation and made logging paths robust.

3) Environment modernization
- requirements.txt updated to latest libraries (NumPy 2.x, pandas 2.2+, etc.).
- Added env/requirements-py313-nightly-cpu.txt for CPU-only PyTorch on Python 3.13.
- Installed nightly CPU-only PyTorch (no GPU) to run training.

4) Code compatibility fixes (NumPy 2.x)
- Replaced deprecated np.Inf with np.inf.
- Replaced -np.inf with float('-inf') in attention/masking code.

5) Training and outputs
- Supervised runs executed for ETTh1, Weather, and Traffic with small hyperparameters for quick turnaround (debug profiles).
- Outputs saved to:
  - Checkpoints: PatchTST_supervised/checkpoints/<setting>/checkpoint.pth
  - Test plots: PatchTST_supervised/test_results/<setting>/*.pdf
  - Predictions: PatchTST_supervised/results/<setting>/pred.npy
  - Logs: PatchTST_supervised/logs/LongForecasting/
  - Metrics summary: PatchTST_supervised/result.txt
  
6)Learning
This setup is Supervised Learning:
- You train a forecasting model (e.g., PatchTST) on labeled time series data (historical inputs → next-step or multi-step future targets).
- The trained model is saved to disk as a checkpoint.
- The app loads that checkpoint to predict future values on unseen CSVs, then compares the predictions against simple baseline methods (last value, mean, zero, random), showing metrics like MAE, RMSE, R².

This is not Self-Supervised Learning (no pretext tasks or unlabeled pretraining implemented here). If you later want self-supervised pretraining, you’ll add a separate pretraining script and fine-tune.

Short summary of PatchTST:
- PatchTST splits each time series into overlapping patches, encodes them with a Transformer encoder (multi-head self-attention), and predicts future windows. It is strong for long-context forecasting and supports channel-wise modeling with optional reversible instance normalization (RevIN).

Conclusion:
The repository is updated and runnable on a modern Windows CPU environment. We verified runs on all three datasets, produced metrics, checkpoints, and result artifacts, and prepared concise scripts and instructions. This setup is ready for demonstration and further experimentation (e.g., different horizons, model sizes, or training epochs).