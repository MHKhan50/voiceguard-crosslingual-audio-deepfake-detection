# VoiceGuard baseline results

| Features + classifier | Condition | EER % [95% CI] | Unseen-attack EER % | AUC | ECE | Brier | Error at 80% coverage |
|---|---|---|---|---|---|---|---|
| mfcc+logreg | Clean | 18.1 [17.3, 18.9] | 18.0 | 0.900 | 0.131 | 0.211 | 26.4% |
| mfcc+logreg | Phone line | 50.6 [49.6, 51.9] | 51.2 | 0.482 | 0.189 | 0.215 | 22.6% |
| mfcc+logreg | Phone + noise 15 dB | 38.6 [37.7, 39.6] | 38.8 | 0.664 | 0.284 | 0.341 | 48.8% |
| mfcc+mlp | Clean | 16.3 [15.5, 17.1] | 16.6 | 0.911 | 0.144 | 0.210 | 24.0% |
| mfcc+mlp | Phone line | 33.8 [32.7, 34.9] | 32.8 | 0.682 | 0.127 | 0.161 | 14.1% |
| mfcc+mlp | Phone + noise 15 dB | 50.0 [49.6, 50.4] | 50.2 | 0.500 | 0.168 | 0.191 | 20.6% |
| wavlm_l12+logreg | Clean | 5.1 [4.7, 5.5] | 3.7 | 0.988 | 0.043 | 0.060 | 2.2% |
| wavlm_l12+logreg | Phone line | 9.6 [8.9, 10.3] | 8.8 | 0.924 | 0.064 | 0.079 | 3.3% |
| wavlm_l12+logreg | Phone + noise 15 dB | 33.5 [32.5, 34.5] | 33.2 | 0.664 | 0.185 | 0.189 | 14.8% |
| wavlm_l12+mlp | Clean | 5.8 [5.2, 6.2] | 3.4 | 0.985 | 0.076 | 0.096 | 5.8% |
| wavlm_l12+mlp | Phone line | 7.1 [6.5, 7.7] | 5.7 | 0.975 | 0.024 | 0.047 | 1.2% |
| wavlm_l12+mlp | Phone + noise 15 dB | 35.9 [34.9, 37.0] | 35.8 | 0.641 | 0.193 | 0.196 | 15.6% |
| wavlm_l3+logreg | Clean | 3.3 [2.9, 3.9] | 3.5 | 0.995 | 0.068 | 0.085 | 4.7% |
| wavlm_l3+logreg | Phone line | 16.8 [15.8, 17.6] | 17.8 | 0.904 | 0.074 | 0.096 | 6.0% |
| wavlm_l3+logreg | Phone + noise 15 dB | 25.5 [24.5, 26.6] | 26.3 | 0.831 | 0.597 | 0.601 | 64.6% |
| wavlm_l3+mlp | Clean | 1.9 [1.6, 2.4] | 1.7 | 0.998 | 0.026 | 0.049 | 1.4% |
| wavlm_l3+mlp | Phone line | 12.0 [11.2, 12.8] | 12.3 | 0.954 | 0.027 | 0.066 | 3.4% |
| wavlm_l3+mlp | Phone + noise 15 dB | 35.2 [34.2, 36.3] | 36.2 | 0.715 | 0.294 | 0.328 | 37.2% |
| wavlm_l6+logreg | Clean | 2.7 [2.2, 3.2] | 2.7 | 0.997 | 0.044 | 0.063 | 2.4% |
| wavlm_l6+logreg | Phone line | 3.3 [2.8, 3.6] | 3.2 | 0.993 | 0.012 | 0.022 | 0.3% |
| wavlm_l6+logreg | Phone + noise 15 dB | 9.8 [9.1, 10.4] | 9.6 | 0.956 | 0.062 | 0.078 | 4.0% |
| wavlm_l6+mlp | Clean | 1.8 [1.5, 2.2] | 1.5 | 0.998 | 0.048 | 0.056 | 1.5% |
| wavlm_l6+mlp | Phone line | 2.0 [1.7, 2.4] | 1.5 | 0.997 | 0.015 | 0.020 | 0.3% |
| wavlm_l6+mlp | Phone + noise 15 dB | 9.3 [8.6, 9.9] | 8.7 | 0.957 | 0.077 | 0.091 | 5.7% |
| wavlm_l9+logreg | Clean | 3.9 [3.5, 4.3] | 2.6 | 0.991 | 0.034 | 0.047 | 1.3% |
| wavlm_l9+logreg | Phone line | 5.1 [4.6, 5.8] | 4.5 | 0.980 | 0.020 | 0.032 | 0.7% |
| wavlm_l9+logreg | Phone + noise 15 dB | 13.0 [12.2, 13.9] | 13.2 | 0.882 | 0.101 | 0.110 | 5.3% |
| wavlm_l9+mlp | Clean | 4.1 [3.7, 4.5] | 2.9 | 0.990 | 0.042 | 0.062 | 2.9% |
| wavlm_l9+mlp | Phone line | 6.4 [5.8, 7.1] | 5.7 | 0.977 | 0.017 | 0.037 | 1.0% |
| wavlm_l9+mlp | Phone + noise 15 dB | 11.7 [10.9, 12.5] | 11.9 | 0.939 | 0.081 | 0.098 | 5.1% |
