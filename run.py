import h5py
from detect_adv import create_detector
from utils import *
import sys
#sys.path.append('../Thesis_CNN_mnist/')
from cnn import MnistCNN
sys.path.append('../Thesis_Utilities/')
from utilities import load_datasets

import tensorflow as tf
import numpy as np
import pickle


# Load MNIST and, if omniglot_bool, Omniglot datasets.
x_train, y_train, _, _, x_test, y_test = load_datasets(test_size=10000, val_size=40000, omniglot_bool=True,
                                                               name_data_set='data_omni.h5', force=False,
                                                               create_file=True, r_seed=None)

# Limit the data to facilitate runs on slower computers.
langd=10000
x_train = x_train[:langd]
y_train = y_train[:langd]
x_test = x_test[9900:10100]  #50/50 mnist and omniglot
y_test = y_test[9900:10100]

# Build model.
tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess, save_dir='MnistCNN_save/')

# Compute kernel density estimations and linear regression model.
#kdes, lr, scaler_dens, scaler_uncerts, scaler_dens2, scaler_uncerts2, lr_robust = \
#    create_detector(net, x_train, y_train, x_test, y_test, dataset='mnist')

# Save models.
filename = "logregmodel.sav"
filename2 = "kdes.sav"
filename3 = "scaler_dens.sav"
filename4 = "scaler_uncerts.sav"

#pickle.dump(lr_robust, open(filename, "wb"))
#pickle.dump(kdes, open(filename2, "wb"))
#pickle.dump(scaler_dens2, open(filename3, "wb"))
#pickle.dump(scaler_uncerts2, open(filename4, "wb"))

loaded_logreg = pickle.load(open(filename, 'rb'))
loaded_kdes = pickle.load(open(filename2, 'rb'))
scaler_dens = pickle.load(open(filename3, 'rb'))
scaler_uncerts = pickle.load(open(filename4, 'rb'))

# Testing model on test images.
acc = 0
prob = []
for i in range(0, len(x_test)):
    test_image = x_test[i:i+1]
    test_label = y_test[i:i+1]
    xpred, _, xact = net.predict(test_image)
    uncerts = get_montecarlo_predictions(net, test_image, num_iter=40).var(axis=0).mean(axis=1)
    hid_acts = xact[-2]
    score = loaded_kdes[xpred[0]].score_samples(np.reshape(hid_acts, (1, -1)))[0]

    # Z-score using StandardScaler-models
    uncerts_z = scaler_uncerts.transform(uncerts.reshape(1, -1))
    score_z = scaler_dens.transform(score.reshape(1, -1))

    values = np.concatenate((score_z.reshape((1, -1)), uncerts_z.reshape((1, -1))), axis=0).transpose([1, 0])

    # Predict using LogisticRegressionCV-model
    prob.append(loaded_logreg.predict(values)[0])

    if np.squeeze(np.nonzero(test_label[0])) < 10:
        true = 0
    else:
        true = 1
    if prob[i] == true:
        acc = acc + 1

print("Accuracy: {}".format(acc/len(x_test)))

# Compute ROC and AUC
n_samples = np.int(len(x_test)/2)

_, _, auc_score = compute_roc(
    probs_neg=prob[:n_samples],
    probs_pos=prob[n_samples:],
    plot=True
)
print('Detector ROC-AUC score: %0.4f' % auc_score)
