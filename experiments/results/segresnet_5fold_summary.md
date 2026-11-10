# SegResNet — 5-Fold Cross-Validation Results

## Training configuration

- Dataset: ISLES-2022
- Valid subjects: 250
- Cross-validation: 5-fold
- Model: SegResNet
- Input channels: DWI + ADC + FLAIR
- Output channels: 1
- Epochs per fold: 30
- Batch size: 1
- Learning rate: 0.001
- ROI size: 96 × 96 × 32
- AMP FP16: Enabled
- Validation subjects per fold: 50
- Training subjects per fold: 200

## Results

| Fold | Best Validation Dice | Training Time |
|---:|---:|---:|
| 0 | 0.5328 | 79.6 min |
| 1 | 0.5847 | 103.5 min |
| 2 | 0.5476 | 90.3 min |
| 3 | 0.4043 | 90.8 min |
| 4 | 0.6822 | 90.6 min |
| **Mean** | **0.5503** | — |
| **Std** | **0.0897** | — |

## Best epochs

- Fold 0: Epoch 25
- Fold 1: See 	rain.log
- Fold 2: See 	rain.log
- Fold 3: See 	rain.log
- Fold 4: See 	rain.log

## Notes

The five folds were trained independently. The best checkpoint from one fold was not used to initialize another fold.

Training checkpoints and raw training logs remain local under experiments/runs/ and are intentionally excluded from Git.
