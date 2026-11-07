# Notes (first baseline run, 7 Nov 2026)

1. Layer 6 of WavLM held up best on the simulated phone line (EER 1.8% clean, 2.0% phone line with the MLP), while
   layer 3 was just as good on clean audio (1.9%) but rose to 12.0%. My guess is that the early layers rely more on
   fine acoustic detail that the 8 kHz band and mu-law coding remove, while layer 6 keeps cues that survive the
   channel. This run does not test that guess; probing what each layer encodes would.

2. With noise added, A18 was the hardest attack for layer 6 + MLP (3.7% clean to 25.6%), followed by A17 (1.1% to
   18.4%) and A19 (4.7% to 14.4%). I have not yet checked how these three systems differ from the others, so I cannot
   say why. A19 uses the same algorithm as A06 in training and was still among the hardest, so having seen an attack in
   training did not protect against the noisy channel.

3. For phone fraud, the abstention result matters most to me. On the phone line, layer 6 + MLP had 2.5% error on all
   clips and 0.3% on the 80% it was most confident about, so the remaining 20% could go to a human check. But this
   only works when the model is unsure in the right places: layer 3 + logistic regression with noise stayed between
   58% and 65% error at every coverage.
