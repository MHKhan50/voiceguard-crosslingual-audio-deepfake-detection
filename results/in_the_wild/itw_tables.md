# In-the-Wild (trained on ASVspoof 2019 LA only)

31779 clips (19963 bona fide, 11816 spoof), first 4 s of each. Mean over classifier seeds 42, 43, 44; sd in brackets.

| Model | EER % | AUC | Cllr | minCllr | ECE |
|---|---:|---:|---:|---:|---:|
| mfcc+logreg | 66.1 (0.0) | 0.290 | 3.77 | 1.00 | 0.572 |
| mfcc+mlp | 56.0 (0.7) | 0.441 | 2.31 | 1.00 | 0.542 |
| wavlm_l12+logreg | 23.0 (0.0) | 0.829 | 3.09 | 0.72 | 0.545 |
| wavlm_l12+mlp | 29.1 (1.0) | 0.742 | 3.30 | 0.83 | 0.533 |
| wavlm_l3+logreg | 32.5 (0.0) | 0.731 | 2.14 | 0.87 | 0.264 |
| wavlm_l3+mlp | 20.7 (3.5) | 0.860 | 0.96 | 0.65 | 0.114 |
| wavlm_l6+logreg | 18.1 (0.0) | 0.895 | 1.33 | 0.59 | 0.184 |
| wavlm_l6+mlp | 18.0 (1.5) | 0.888 | 1.38 | 0.59 | 0.162 |
| wavlm_l9+logreg | 24.6 (0.0) | 0.823 | 2.56 | 0.74 | 0.394 |
| wavlm_l9+mlp | 25.4 (0.4) | 0.807 | 2.22 | 0.77 | 0.349 |

## Threshold fixed on ASVspoof dev (seed 42)

| Model | Real called fake % | Fake called real % | HTER % |
|---|---:|---:|---:|
| mfcc+logreg | 98.1 | 4.3 | 51.2 |
| mfcc+mlp | 99.1 | 2.0 | 50.5 |
| wavlm_l12+logreg | 93.1 | 0.2 | 46.6 |
| wavlm_l12+mlp | 92.0 | 0.4 | 46.2 |
| wavlm_l3+logreg | 39.3 | 24.7 | 32.0 |
| wavlm_l3+mlp | 14.3 | 21.7 | 18.0 |
| wavlm_l6+logreg | 35.8 | 7.3 | 21.5 |
| wavlm_l6+mlp | 30.7 | 7.0 | 18.8 |
| wavlm_l9+logreg | 71.6 | 2.2 | 36.9 |
| wavlm_l9+mlp | 70.3 | 4.5 | 37.4 |

## Demo rule (fake if p >= 0.95, real if p <= 0.05, even prior; seed 42)

| Model | Real: fake % | Real: uncertain % | Real: real % | Fake: fake % | Fake: uncertain % | Fake: real % |
|---|---:|---:|---:|---:|---:|---:|
| mfcc+logreg | 83.5 | 16.2 | 0.4 | 65.9 | 33.3 | 0.7 |
| mfcc+mlp | 92.4 | 7.6 | 0.0 | 87.2 | 12.6 | 0.2 |
| wavlm_l12+logreg | 84.2 | 12.7 | 3.2 | 98.9 | 1.0 | 0.1 |
| wavlm_l12+mlp | 77.4 | 19.9 | 2.7 | 97.5 | 2.4 | 0.1 |
| wavlm_l3+logreg | 26.6 | 31.4 | 42.0 | 59.2 | 29.5 | 11.3 |
| wavlm_l3+mlp | 4.8 | 30.3 | 64.8 | 51.3 | 43.0 | 5.7 |
| wavlm_l6+logreg | 21.6 | 38.4 | 40.0 | 85.1 | 12.3 | 2.6 |
| wavlm_l6+mlp | 15.7 | 34.6 | 49.7 | 83.2 | 13.8 | 2.9 |
| wavlm_l9+logreg | 56.3 | 27.3 | 16.4 | 94.1 | 5.2 | 0.7 |
| wavlm_l9+mlp | 44.4 | 45.3 | 10.3 | 87.2 | 11.9 | 0.9 |
