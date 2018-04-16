# Implementation of "Adversarial Detection from Artifacts" for Novelty Detection
This is work for the Master's thesis project "Input Verification of Deep Neural Networks". The algorithm's purpose is to reject novelty images when trained on Mnist and tuned on adversarial images.

### Requirements
Numpy, h5py, pickle, sklearn, matplotlib, multiprocessing.

## Usage
### Compute kernel density estimations and linear regression model
To generate kernel density models for each class, and the logistic regression model for inference, call the `create_detector` function as below. The function also returns StandardScaler and RobustScaler models for densities and uncertainties which enables two different methods of scaling test data. As of now, the different scalers don't show much difference in performance.
```
kdes, lr, scaler_dens, scaler_uncerts, scaler_dens2, scaler_uncerts2, lr_robust = create_detector(net, x_train, y_train, x_test, y_test, dataset='mnist')
```

### Evaluate
Given a `test_image` to try the model on, the hidden layer (last dropout layer) activations of the network are needed. These activations are fed to the kernel densities to be get each class kernel density score. By enabling the dropouts to alter between runs, one can find the network uncertainty on the image (mean of the variance). 
```
xpred, _, xact = net.predict(test_image)
hid_acts = xact[-2]
score = loaded_kdes[xpred[0]].score_samples(np.reshape(hid_acts, (1, -1)))[0]
uncerts = get_montecarlo_predictions(net, test_image, num_iter=40).var(axis=0).mean(axis=1)
```

Using the saved StandardScaler or RobustScaler model, z-score the input image uncertainty and density score. 
```
uncerts_z = scaler_uncerts.transform(uncerts.reshape(1, -1))
score_z = scaler_dens.transform(score.reshape(1, -1))

values = np.concatenate((score_z.reshape((1, -1)), uncerts_z.reshape((1, -1))), axis=0).transpose([1, 0])
```

It is now possible to compute the probability of the input image belonging to the neural network learned domain or not by calling the logistic regression model.
```
prob.append(loaded_logreg.predict(values)[0])
```

## Acknowledgments
The work is an implementation of the algorithm in "Detecting Adversarial Samples from Artifacts" (Feinman et al., 2017). 

**Reference:** https://arxiv.org/abs/1703.00410

The programming is based upon the corresponding git https://github.com/rfeinman/detecting-adversarial-samples.



