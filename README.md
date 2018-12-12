claimrank-allennlp
===

This repository contains the code used to train the claim selection models, as well as the e2e post-modifier generator models for TO BE PUBLISHED.


To train models run:
```{bash}
allennlp train experiments/MODEL_CONFIG -s RESULTS_DIRECTORY --include-package claimrank```
```

To make predictions run:
```{bash}
allennlp predict PATH_TO_TEST_DATA --predictor seq2seq-claimrank --include-package claimrank
```
