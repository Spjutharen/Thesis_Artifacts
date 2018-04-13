# Implementation of "Adversarial Detection from Artifacts" for Novelty Detection
This is work for the Master's thesis project "Input Verification of Deep Neural Networks".

This is an implementation of the algorithm in "Detecting Adversarial Samples from Artifacts" (https://arxiv.org/abs/1703.00410) (Feinman et al., 2017). Instead of focusing on the detection of adversarial images, we try to detect novelty images (Omniglot) when fed to a CNN network trained on Mnist. The work is based upon https://github.com/rfeinman/detecting-adversarial-samples.

### Requirements
Numpy, h5py, pickle, sklearn, matplotlib, multiprocessing

## Usage
### Compute kernel density estimations and linear regression model

```
kdes, lr, scaler_dens, scaler_uncerts, scaler_dens2, scaler_uncerts2, lr_robust = create_detector(net, x_train, y_train, x_test, y_test, dataset='mnist')
```
