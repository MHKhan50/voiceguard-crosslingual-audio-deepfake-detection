# VoiceGuard baseline results

| Features + classifier | Condition | EER % [95% CI] | Unseen-attack EER % | AUC | ECE | Brier | Error at 80% coverage |
|---|---|---|---|---|---|---|---|
| mfcc+logreg | Clean | 18.1 [17.3, 18.9] | 18.0 | 0.900 | 0.012 | 0.121 | 11.3% |
| mfcc+logreg | Phone line | 50.6 [49.6, 51.9] | 51.2 | 0.482 | 0.199 | 0.208 | 22.5% |
| mfcc+logreg | Phone + noise 15 dB | 38.6 [37.7, 39.6] | 38.8 | 0.664 | 0.120 | 0.225 | 31.3% |
| mfcc+mlp | Clean | 16.3 [15.5, 17.1] | 16.6 | 0.911 | 0.052 | 0.131 | 12.5% |
| mfcc+mlp | Phone line | 33.8 [32.7, 34.9] | 32.8 | 0.682 | 0.180 | 0.185 | 14.1% |
| mfcc+mlp | Phone + noise 15 dB | 50.0 [49.6, 50.4] | 50.2 | 0.500 | 0.194 | 0.200 | 20.6% |
| wavlm_l12+logreg | Clean | 5.1 [4.7, 5.5] | 3.7 | 0.988 | 0.030 | 0.044 | 1.2% |
| wavlm_l12+logreg | Phone line | 9.6 [8.9, 10.3] | 8.8 | 0.924 | 0.087 | 0.097 | 3.9% |
| wavlm_l12+logreg | Phone + noise 15 dB | 33.5 [32.5, 34.5] | 33.2 | 0.664 | 0.195 | 0.196 | 14.8% |
| wavlm_l12+mlp | Clean | 5.8 [5.2, 6.2] | 3.4 | 0.985 | 0.059 | 0.073 | 2.9% |
| wavlm_l12+mlp | Phone line | 7.1 [6.5, 7.7] | 5.7 | 0.975 | 0.039 | 0.057 | 1.5% |
| wavlm_l12+mlp | Phone + noise 15 dB | 35.9 [34.9, 37.0] | 35.8 | 0.641 | 0.202 | 0.202 | 15.5% |
| wavlm_l3+logreg | Clean | 3.3 [2.9, 3.9] | 3.5 | 0.995 | 0.048 | 0.061 | 2.4% |
| wavlm_l3+logreg | Phone line | 16.8 [15.8, 17.6] | 17.8 | 0.904 | 0.080 | 0.097 | 5.9% |
| wavlm_l3+logreg | Phone + noise 15 dB | 25.5 [24.5, 26.6] | 26.3 | 0.831 | 0.542 | 0.550 | 60.0% |
| wavlm_l3+mlp | Clean | 1.9 [1.6, 2.4] | 1.7 | 0.998 | 0.010 | 0.027 | 0.4% |
| wavlm_l3+mlp | Phone line | 12.0 [11.2, 12.8] | 12.3 | 0.954 | 0.035 | 0.064 | 3.2% |
| wavlm_l3+mlp | Phone + noise 15 dB | 35.2 [34.2, 36.3] | 36.2 | 0.715 | 0.245 | 0.279 | 30.9% |
| wavlm_l6+logreg | Clean | 2.7 [2.2, 3.2] | 2.7 | 0.997 | 0.024 | 0.039 | 0.8% |
| wavlm_l6+logreg | Phone line | 3.3 [2.8, 3.6] | 3.2 | 0.993 | 0.018 | 0.029 | 0.4% |
| wavlm_l6+logreg | Phone + noise 15 dB | 9.8 [9.1, 10.4] | 9.6 | 0.956 | 0.058 | 0.073 | 3.4% |
| wavlm_l6+mlp | Clean | 1.8 [1.5, 2.2] | 1.5 | 0.998 | 0.034 | 0.041 | 0.7% |
| wavlm_l6+mlp | Phone line | 2.0 [1.7, 2.4] | 1.5 | 0.997 | 0.010 | 0.016 | 0.2% |
| wavlm_l6+mlp | Phone + noise 15 dB | 9.3 [8.6, 9.9] | 8.7 | 0.957 | 0.066 | 0.080 | 4.5% |
| wavlm_l9+logreg | Clean | 3.9 [3.5, 4.3] | 2.6 | 0.991 | 0.024 | 0.035 | 0.7% |
| wavlm_l9+logreg | Phone line | 5.1 [4.6, 5.8] | 4.5 | 0.980 | 0.028 | 0.038 | 0.8% |
| wavlm_l9+logreg | Phone + noise 15 dB | 13.0 [12.2, 13.9] | 13.2 | 0.882 | 0.120 | 0.126 | 5.6% |
| wavlm_l9+mlp | Clean | 4.1 [3.7, 4.5] | 2.9 | 0.990 | 0.028 | 0.043 | 1.3% |
| wavlm_l9+mlp | Phone line | 6.4 [5.8, 7.1] | 5.7 | 0.977 | 0.025 | 0.040 | 1.2% |
| wavlm_l9+mlp | Phone + noise 15 dB | 11.7 [10.9, 12.5] | 11.9 | 0.939 | 0.126 | 0.131 | 5.4% |

## Calibration

Cllr and minCllr in bits (lower is better) do not depend on the class mix. ECE columns: raw classifier output, Platt fitted on dev (v0.2 method), and the LLR calibrator at the eval sample's spoof share.

| Features + classifier | Condition | Cllr | minCllr | Calibration loss | ECE raw | ECE Platt (dev mix) | ECE LLR at eval prior |
|---|---|---|---|---|---|---|---|
| mfcc+logreg | Clean | 0.676 | 0.560 | 0.116 | 0.152 | 0.131 | 0.012 |
| mfcc+logreg | Phone line | 3.459 | 0.974 | 2.485 | 0.210 | 0.189 | 0.199 |
| mfcc+logreg | Phone + noise 15 dB | 1.137 | 0.890 | 0.246 | 0.312 | 0.284 | 0.120 |
| mfcc+mlp | Clean | 0.705 | 0.527 | 0.178 | 0.120 | 0.144 | 0.052 |
| mfcc+mlp | Phone line | 1.842 | 0.880 | 0.961 | 0.202 | 0.127 | 0.180 |
| mfcc+mlp | Phone + noise 15 dB | 2.402 | 0.996 | 1.406 | 0.204 | 0.168 | 0.194 |
| wavlm_l12+logreg | Clean | 0.267 | 0.189 | 0.078 | 0.031 | 0.043 | 0.030 |
| wavlm_l12+logreg | Phone line | 1.094 | 0.317 | 0.777 | 0.124 | 0.064 | 0.087 |
| wavlm_l12+logreg | Phone + noise 15 dB | 3.252 | 0.497 | 2.754 | 0.203 | 0.185 | 0.195 |
| wavlm_l12+mlp | Clean | 0.435 | 0.205 | 0.230 | 0.031 | 0.076 | 0.059 |
| wavlm_l12+mlp | Phone line | 0.433 | 0.261 | 0.172 | 0.119 | 0.024 | 0.039 |
| wavlm_l12+mlp | Phone + noise 15 dB | 3.265 | 0.718 | 2.547 | 0.205 | 0.193 | 0.202 |
| wavlm_l3+logreg | Clean | 0.371 | 0.123 | 0.248 | 0.027 | 0.068 | 0.048 |
| wavlm_l3+logreg | Phone line | 1.121 | 0.543 | 0.577 | 0.107 | 0.074 | 0.080 |
| wavlm_l3+logreg | Phone + noise 15 dB | 4.433 | 0.710 | 3.723 | 0.438 | 0.597 | 0.542 |
| wavlm_l3+mlp | Clean | 0.170 | 0.072 | 0.098 | 0.001 | 0.026 | 0.010 |
| wavlm_l3+mlp | Phone line | 0.499 | 0.386 | 0.113 | 0.083 | 0.027 | 0.035 |
| wavlm_l3+mlp | Phone + noise 15 dB | 1.667 | 0.858 | 0.809 | 0.178 | 0.294 | 0.245 |
| wavlm_l6+logreg | Clean | 0.226 | 0.099 | 0.128 | 0.016 | 0.044 | 0.024 |
| wavlm_l6+logreg | Phone line | 0.228 | 0.128 | 0.099 | 0.040 | 0.012 | 0.018 |
| wavlm_l6+logreg | Phone + noise 15 dB | 0.548 | 0.345 | 0.203 | 0.070 | 0.062 | 0.058 |
| wavlm_l6+mlp | Clean | 0.263 | 0.066 | 0.197 | 0.004 | 0.048 | 0.034 |
| wavlm_l6+mlp | Phone line | 0.118 | 0.081 | 0.036 | 0.018 | 0.015 | 0.010 |
| wavlm_l6+mlp | Phone + noise 15 dB | 0.685 | 0.340 | 0.345 | 0.082 | 0.077 | 0.066 |
| wavlm_l9+logreg | Clean | 0.222 | 0.148 | 0.074 | 0.023 | 0.034 | 0.024 |
| wavlm_l9+logreg | Phone line | 0.346 | 0.193 | 0.153 | 0.053 | 0.020 | 0.028 |
| wavlm_l9+logreg | Phone + noise 15 dB | 1.638 | 0.401 | 1.237 | 0.150 | 0.101 | 0.120 |
| wavlm_l9+mlp | Clean | 0.270 | 0.166 | 0.104 | 0.016 | 0.042 | 0.028 |
| wavlm_l9+mlp | Phone line | 0.365 | 0.253 | 0.112 | 0.072 | 0.017 | 0.025 |
| wavlm_l9+mlp | Phone + noise 15 dB | 1.357 | 0.431 | 0.926 | 0.191 | 0.081 | 0.126 |
