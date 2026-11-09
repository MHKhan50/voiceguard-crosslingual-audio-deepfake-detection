# VoiceGuard baseline results

| Features + classifier | Condition | EER % [95% CI] | Unseen-attack EER % | AUC | ECE | Brier | Error at 80% coverage |
|---|---|---|---|---|---|---|---|
| mfcc+logreg | Clean | 19.7 [19.0, 20.6] | 20.4 | 0.878 | 0.028 | 0.137 | 13.9% |
| mfcc+logreg | Phone line | 50.3 [48.9, 51.7] | 50.5 | 0.491 | 0.198 | 0.208 | 22.3% |
| mfcc+logreg | Phone + noise 15 dB | 37.3 [36.4, 38.4] | 37.2 | 0.676 | 0.217 | 0.297 | 43.1% |
| mfcc+logreg | Opus 16 kbps | 15.3 [14.5, 16.0] | 13.3 | 0.928 | 0.180 | 0.177 | 7.8% |
| mfcc+mlp | Clean | 15.0 [14.3, 16.0] | 15.8 | 0.922 | 0.063 | 0.129 | 11.9% |
| mfcc+mlp | Phone line | 40.3 [39.5, 41.2] | 40.1 | 0.597 | 0.193 | 0.198 | 17.2% |
| mfcc+mlp | Phone + noise 15 dB | 50.0 [49.9, 50.0] | 50.0 | 0.500 | 0.195 | 0.200 | 20.7% |
| mfcc+mlp | Opus 16 kbps | 12.6 [11.8, 13.2] | 10.9 | 0.921 | 0.134 | 0.141 | 6.8% |
| wavlm_l12+logreg | Clean | 5.2 [4.7, 5.8] | 3.7 | 0.986 | 0.031 | 0.045 | 1.2% |
| wavlm_l12+logreg | Phone line | 13.2 [12.3, 14.1] | 12.6 | 0.880 | 0.097 | 0.106 | 5.3% |
| wavlm_l12+logreg | Phone + noise 15 dB | 39.3 [38.4, 40.3] | 39.3 | 0.607 | 0.198 | 0.198 | 17.0% |
| wavlm_l12+logreg | Opus 16 kbps | 4.6 [4.3, 5.2] | 3.4 | 0.988 | 0.020 | 0.033 | 0.4% |
| wavlm_l12+mlp | Clean | 5.6 [5.1, 6.1] | 3.8 | 0.985 | 0.058 | 0.074 | 3.3% |
| wavlm_l12+mlp | Phone line | 6.4 [5.8, 6.9] | 5.3 | 0.977 | 0.045 | 0.059 | 1.4% |
| wavlm_l12+mlp | Phone + noise 15 dB | 32.6 [31.5, 33.7] | 32.5 | 0.675 | 0.201 | 0.201 | 14.4% |
| wavlm_l12+mlp | Opus 16 kbps | 5.7 [5.1, 6.2] | 4.0 | 0.986 | 0.047 | 0.062 | 2.1% |
| wavlm_l3+logreg | Clean | 3.7 [3.2, 4.2] | 3.9 | 0.994 | 0.060 | 0.077 | 4.0% |
| wavlm_l3+logreg | Phone line | 13.5 [12.5, 14.4] | 14.3 | 0.942 | 0.057 | 0.082 | 4.6% |
| wavlm_l3+logreg | Phone + noise 15 dB | 24.2 [23.3, 25.2] | 26.2 | 0.829 | 0.449 | 0.458 | 49.9% |
| wavlm_l3+logreg | Opus 16 kbps | 4.7 [4.2, 5.3] | 5.1 | 0.990 | 0.031 | 0.046 | 1.3% |
| wavlm_l3+mlp | Clean | 1.8 [1.4, 2.3] | 1.9 | 0.998 | 0.011 | 0.025 | 0.3% |
| wavlm_l3+mlp | Phone line | 9.2 [8.5, 10.2] | 9.5 | 0.972 | 0.018 | 0.047 | 1.8% |
| wavlm_l3+mlp | Phone + noise 15 dB | 24.6 [23.6, 25.7] | 25.6 | 0.833 | 0.123 | 0.181 | 19.1% |
| wavlm_l3+mlp | Opus 16 kbps | 2.3 [1.9, 2.8] | 2.2 | 0.997 | 0.004 | 0.013 | 0.2% |
| wavlm_l6+logreg | Clean | 2.6 [2.2, 3.1] | 2.7 | 0.997 | 0.022 | 0.036 | 0.6% |
| wavlm_l6+logreg | Phone line | 3.2 [2.7, 3.6] | 2.8 | 0.992 | 0.024 | 0.034 | 0.4% |
| wavlm_l6+logreg | Phone + noise 15 dB | 10.5 [9.8, 11.2] | 9.9 | 0.949 | 0.069 | 0.082 | 4.0% |
| wavlm_l6+logreg | Opus 16 kbps | 3.3 [2.8, 3.7] | 3.1 | 0.996 | 0.008 | 0.019 | 0.1% |
| wavlm_l6+mlp | Clean | 2.1 [1.8, 2.4] | 1.8 | 0.998 | 0.031 | 0.040 | 0.7% |
| wavlm_l6+mlp | Phone line | 2.2 [1.8, 2.5] | 1.5 | 0.996 | 0.010 | 0.015 | 0.3% |
| wavlm_l6+mlp | Phone + noise 15 dB | 9.5 [8.8, 10.2] | 8.6 | 0.956 | 0.060 | 0.076 | 4.1% |
| wavlm_l6+mlp | Opus 16 kbps | 2.4 [2.1, 3.0] | 2.2 | 0.997 | 0.013 | 0.020 | 0.1% |
| wavlm_l9+logreg | Clean | 3.8 [3.4, 4.2] | 2.7 | 0.992 | 0.024 | 0.033 | 0.6% |
| wavlm_l9+logreg | Phone line | 4.6 [4.1, 5.3] | 4.1 | 0.983 | 0.029 | 0.038 | 0.7% |
| wavlm_l9+logreg | Phone + noise 15 dB | 12.2 [11.3, 13.2] | 12.3 | 0.889 | 0.116 | 0.122 | 5.1% |
| wavlm_l9+logreg | Opus 16 kbps | 4.0 [3.6, 4.4] | 2.8 | 0.991 | 0.019 | 0.027 | 0.3% |
| wavlm_l9+mlp | Clean | 3.4 [3.0, 3.9] | 2.0 | 0.993 | 0.029 | 0.041 | 1.2% |
| wavlm_l9+mlp | Phone line | 5.7 [5.1, 6.3] | 4.9 | 0.982 | 0.027 | 0.041 | 1.0% |
| wavlm_l9+mlp | Phone + noise 15 dB | 9.5 [8.9, 10.3] | 9.5 | 0.963 | 0.115 | 0.122 | 4.8% |
| wavlm_l9+mlp | Opus 16 kbps | 3.6 [3.2, 4.0] | 2.4 | 0.993 | 0.020 | 0.031 | 0.4% |

## Calibration

Cllr and minCllr in bits (lower is better) do not depend on the class mix. ECE columns: raw classifier output, Platt fitted on dev (v0.2 method), and the LLR calibrator at the eval sample's spoof share.

| Features + classifier | Condition | Cllr | minCllr | Calibration loss | ECE raw | ECE Platt (dev mix) | ECE LLR at eval prior |
|---|---|---|---|---|---|---|---|
| mfcc+logreg | Clean | 0.743 | 0.607 | 0.136 | 0.197 | 0.153 | 0.028 |
| mfcc+logreg | Phone line | 3.293 | 0.976 | 2.317 | 0.215 | 0.188 | 0.198 |
| mfcc+logreg | Phone + noise 15 dB | 1.404 | 0.889 | 0.515 | 0.469 | 0.414 | 0.217 |
| mfcc+logreg | Opus 16 kbps | 1.660 | 0.469 | 1.191 | 0.179 | 0.143 | 0.180 |
| mfcc+mlp | Clean | 0.687 | 0.491 | 0.197 | 0.114 | 0.138 | 0.063 |
| mfcc+mlp | Phone line | 2.305 | 0.893 | 1.412 | 0.204 | 0.165 | 0.193 |
| mfcc+mlp | Phone + noise 15 dB | 2.458 | 1.000 | 1.458 | 0.204 | 0.170 | 0.195 |
| mfcc+mlp | Opus 16 kbps | 1.010 | 0.421 | 0.589 | 0.181 | 0.051 | 0.134 |
| wavlm_l12+logreg | Clean | 0.278 | 0.193 | 0.085 | 0.031 | 0.043 | 0.031 |
| wavlm_l12+logreg | Phone line | 1.287 | 0.375 | 0.912 | 0.133 | 0.077 | 0.097 |
| wavlm_l12+logreg | Phone + noise 15 dB | 3.260 | 0.435 | 2.825 | 0.204 | 0.189 | 0.198 |
| wavlm_l12+logreg | Opus 16 kbps | 0.210 | 0.174 | 0.036 | 0.031 | 0.024 | 0.020 |
| wavlm_l12+mlp | Clean | 0.433 | 0.206 | 0.227 | 0.027 | 0.078 | 0.058 |
| wavlm_l12+mlp | Phone line | 0.458 | 0.243 | 0.214 | 0.127 | 0.025 | 0.045 |
| wavlm_l12+mlp | Phone + noise 15 dB | 3.486 | 0.643 | 2.843 | 0.204 | 0.194 | 0.201 |
| wavlm_l12+mlp | Opus 16 kbps | 0.357 | 0.200 | 0.157 | 0.028 | 0.066 | 0.047 |
| wavlm_l3+logreg | Clean | 0.475 | 0.135 | 0.340 | 0.038 | 0.088 | 0.060 |
| wavlm_l3+logreg | Phone line | 0.614 | 0.435 | 0.179 | 0.063 | 0.075 | 0.057 |
| wavlm_l3+logreg | Phone + noise 15 dB | 3.362 | 0.708 | 2.654 | 0.333 | 0.505 | 0.449 |
| wavlm_l3+logreg | Opus 16 kbps | 0.286 | 0.174 | 0.112 | 0.021 | 0.048 | 0.031 |
| wavlm_l3+mlp | Clean | 0.153 | 0.065 | 0.088 | 0.004 | 0.023 | 0.011 |
| wavlm_l3+mlp | Phone line | 0.343 | 0.298 | 0.045 | 0.062 | 0.014 | 0.018 |
| wavlm_l3+mlp | Phone + noise 15 dB | 0.974 | 0.702 | 0.272 | 0.099 | 0.194 | 0.123 |
| wavlm_l3+mlp | Opus 16 kbps | 0.097 | 0.088 | 0.009 | 0.010 | 0.004 | 0.004 |
| wavlm_l6+logreg | Clean | 0.205 | 0.097 | 0.109 | 0.017 | 0.038 | 0.022 |
| wavlm_l6+logreg | Phone line | 0.273 | 0.121 | 0.152 | 0.045 | 0.015 | 0.024 |
| wavlm_l6+logreg | Phone + noise 15 dB | 0.687 | 0.351 | 0.336 | 0.080 | 0.072 | 0.069 |
| wavlm_l6+logreg | Opus 16 kbps | 0.126 | 0.112 | 0.013 | 0.013 | 0.013 | 0.008 |
| wavlm_l6+mlp | Clean | 0.244 | 0.076 | 0.168 | 0.003 | 0.046 | 0.031 |
| wavlm_l6+mlp | Phone line | 0.123 | 0.088 | 0.035 | 0.020 | 0.012 | 0.010 |
| wavlm_l6+mlp | Phone + noise 15 dB | 0.625 | 0.337 | 0.288 | 0.073 | 0.072 | 0.060 |
| wavlm_l6+mlp | Opus 16 kbps | 0.126 | 0.091 | 0.035 | 0.010 | 0.019 | 0.013 |
| wavlm_l9+logreg | Clean | 0.214 | 0.143 | 0.071 | 0.023 | 0.034 | 0.024 |
| wavlm_l9+logreg | Phone line | 0.354 | 0.176 | 0.178 | 0.052 | 0.023 | 0.029 |
| wavlm_l9+logreg | Phone + noise 15 dB | 1.697 | 0.371 | 1.326 | 0.146 | 0.099 | 0.116 |
| wavlm_l9+logreg | Opus 16 kbps | 0.190 | 0.149 | 0.041 | 0.028 | 0.020 | 0.019 |
| wavlm_l9+mlp | Clean | 0.243 | 0.135 | 0.108 | 0.016 | 0.042 | 0.029 |
| wavlm_l9+mlp | Phone line | 0.377 | 0.222 | 0.156 | 0.071 | 0.020 | 0.027 |
| wavlm_l9+mlp | Phone + noise 15 dB | 1.194 | 0.336 | 0.858 | 0.182 | 0.079 | 0.115 |
| wavlm_l9+mlp | Opus 16 kbps | 0.182 | 0.139 | 0.043 | 0.017 | 0.031 | 0.020 |

## Classifier seeds

Same clips and features; only the classifier seed changes (42, 43, 44). The tables above use seed 42. sd is the sample standard deviation over seeds. Logistic regression is fitted deterministically, so its sd is expected to be zero; the MLP seed sets its initial weights and early-stopping split.

| Features + classifier | Condition | EER % mean (sd) | Cllr mean (sd) | minCllr mean (sd) | ECE mean (sd) |
|---|---|---|---|---|---|
| mfcc+logreg | Clean | 19.71 (0.00) | 0.743 (0.000) | 0.607 (0.000) | 0.028 (0.000) |
| mfcc+logreg | Phone line | 50.31 (0.00) | 3.293 (0.000) | 0.976 (0.000) | 0.198 (0.000) |
| mfcc+logreg | Phone + noise 15 dB | 37.34 (0.00) | 1.404 (0.000) | 0.889 (0.000) | 0.217 (0.000) |
| mfcc+logreg | Opus 16 kbps | 15.26 (0.00) | 1.660 (0.000) | 0.469 (0.000) | 0.180 (0.000) |
| mfcc+mlp | Clean | 16.25 (1.10) | 0.742 (0.048) | 0.520 (0.026) | 0.073 (0.008) |
| mfcc+mlp | Phone line | 40.56 (3.53) | 2.129 (0.335) | 0.929 (0.038) | 0.185 (0.014) |
| mfcc+mlp | Phone + noise 15 dB | 50.84 (0.76) | 2.382 (0.141) | 1.000 (0.000) | 0.194 (0.002) |
| mfcc+mlp | Opus 16 kbps | 13.38 (0.73) | 0.973 (0.056) | 0.447 (0.023) | 0.128 (0.008) |
| wavlm_l12+logreg | Clean | 5.20 (0.00) | 0.278 (0.000) | 0.193 (0.000) | 0.031 (0.000) |
| wavlm_l12+logreg | Phone line | 13.15 (0.00) | 1.287 (0.000) | 0.375 (0.000) | 0.097 (0.000) |
| wavlm_l12+logreg | Phone + noise 15 dB | 39.33 (0.00) | 3.260 (0.000) | 0.435 (0.000) | 0.198 (0.000) |
| wavlm_l12+logreg | Opus 16 kbps | 4.64 (0.00) | 0.210 (0.000) | 0.174 (0.000) | 0.020 (0.000) |
| wavlm_l12+mlp | Clean | 5.61 (0.14) | 0.419 (0.015) | 0.205 (0.005) | 0.055 (0.002) |
| wavlm_l12+mlp | Phone line | 5.96 (0.40) | 0.404 (0.066) | 0.227 (0.014) | 0.040 (0.008) |
| wavlm_l12+mlp | Phone + noise 15 dB | 27.02 (4.95) | 3.593 (0.098) | 0.573 (0.062) | 0.201 (0.000) |
| wavlm_l12+mlp | Opus 16 kbps | 5.62 (0.09) | 0.354 (0.010) | 0.202 (0.005) | 0.046 (0.001) |
| wavlm_l3+logreg | Clean | 3.68 (0.00) | 0.475 (0.000) | 0.135 (0.000) | 0.060 (0.000) |
| wavlm_l3+logreg | Phone line | 13.48 (0.00) | 0.614 (0.000) | 0.435 (0.000) | 0.057 (0.000) |
| wavlm_l3+logreg | Phone + noise 15 dB | 24.21 (0.00) | 3.362 (0.000) | 0.708 (0.000) | 0.449 (0.000) |
| wavlm_l3+logreg | Opus 16 kbps | 4.69 (0.00) | 0.286 (0.000) | 0.174 (0.000) | 0.031 (0.000) |
| wavlm_l3+mlp | Clean | 2.08 (0.42) | 0.175 (0.028) | 0.079 (0.018) | 0.011 (0.002) |
| wavlm_l3+mlp | Phone line | 6.88 (2.21) | 0.273 (0.074) | 0.235 (0.060) | 0.013 (0.006) |
| wavlm_l3+mlp | Phone + noise 15 dB | 23.58 (1.34) | 1.476 (0.608) | 0.686 (0.022) | 0.132 (0.035) |
| wavlm_l3+mlp | Opus 16 kbps | 2.45 (0.35) | 0.113 (0.019) | 0.098 (0.014) | 0.005 (0.000) |
| wavlm_l6+logreg | Clean | 2.65 (0.00) | 0.205 (0.000) | 0.097 (0.000) | 0.022 (0.000) |
| wavlm_l6+logreg | Phone line | 3.15 (0.00) | 0.273 (0.000) | 0.121 (0.000) | 0.024 (0.000) |
| wavlm_l6+logreg | Phone + noise 15 dB | 10.51 (0.00) | 0.687 (0.000) | 0.351 (0.000) | 0.069 (0.000) |
| wavlm_l6+logreg | Opus 16 kbps | 3.30 (0.00) | 0.126 (0.000) | 0.112 (0.000) | 0.008 (0.000) |
| wavlm_l6+mlp | Clean | 1.71 (0.32) | 0.205 (0.036) | 0.066 (0.009) | 0.025 (0.006) |
| wavlm_l6+mlp | Phone line | 2.17 (0.11) | 0.126 (0.006) | 0.087 (0.002) | 0.010 (0.001) |
| wavlm_l6+mlp | Phone + noise 15 dB | 8.12 (1.37) | 0.699 (0.114) | 0.301 (0.043) | 0.064 (0.011) |
| wavlm_l6+mlp | Opus 16 kbps | 2.11 (0.30) | 0.111 (0.015) | 0.081 (0.009) | 0.010 (0.003) |
| wavlm_l9+logreg | Clean | 3.80 (0.00) | 0.214 (0.000) | 0.143 (0.000) | 0.024 (0.000) |
| wavlm_l9+logreg | Phone line | 4.61 (0.00) | 0.354 (0.000) | 0.176 (0.000) | 0.029 (0.000) |
| wavlm_l9+logreg | Phone + noise 15 dB | 12.24 (0.00) | 1.697 (0.000) | 0.371 (0.000) | 0.116 (0.000) |
| wavlm_l9+logreg | Opus 16 kbps | 3.96 (0.00) | 0.190 (0.000) | 0.149 (0.000) | 0.019 (0.000) |
| wavlm_l9+mlp | Clean | 3.53 (0.13) | 0.254 (0.011) | 0.138 (0.003) | 0.030 (0.001) |
| wavlm_l9+mlp | Phone line | 5.77 (0.29) | 0.361 (0.050) | 0.224 (0.008) | 0.027 (0.004) |
| wavlm_l9+mlp | Phone + noise 15 dB | 8.91 (0.80) | 0.981 (0.253) | 0.318 (0.026) | 0.089 (0.025) |
| wavlm_l9+mlp | Opus 16 kbps | 3.68 (0.11) | 0.189 (0.007) | 0.141 (0.002) | 0.021 (0.001) |
