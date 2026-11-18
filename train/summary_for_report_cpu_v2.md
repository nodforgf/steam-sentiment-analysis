# Steam Sentiment - CPU Training v2

## Dataset
- Total: 2898
- Train / Validation / Test: 2318 / 290 / 290
- Labels: 0 = negative, 1 = positive

## Training configuration
- Device: CPU (2 threads)
- Epochs: 3
- Batch size: 4
- Gradient accumulation: 2
- Effective batch size: 8
- Max length: 128
- Learning rate: 1.5e-05
- Training time: 82.19 minutes

## Model comparison
```
                          model  accuracy  macro_f1
          Baseline (pretrained)  0.848276  0.847689
Fine-tuned old (historical log)  0.830000  0.830000
           Fine-tuned new (CPU)  0.896552  0.896429
```

## Important notes
- New model saved separately; old ./steam_model folder was not modified by this script.
- Hugging Face upload is disabled.
- Neutral is confidence-based, not a trained third class.
- Use validation threshold results to choose the neutral threshold.
- Do not select a threshold based on test-set results.
