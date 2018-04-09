import h5py
from detect_adv import *
from utils import *
import sys
sys.path.append('../Thesis_CNN_mnist/')
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

# Build model.
tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/MnistCNN_save/')

# Compute kernel density estimations and linear regression model.
#kdes, lr, scaler_dens, scaler_uncerts = create_detector(net, x_train, y_train, x_test, y_test, dataset='mnist')

filename = "logregmodel.sav"
filename2 = "kdes.sav"
filename3 = "scaler_dens.sav"
filename4 = "scaler_uncerts.sav"
# pickle.dump(lr, open(filename, "wb"))
# pickle.dump(kdes, open(filename2, "wb"))
# pickle.dump(scaler_dens, open(filename3, "wb"))
# pickle.dump(scaler_uncerts, open(filename4, "wb"))

loaded_logreg = pickle.load(open(filename, 'rb'))
loaded_kdes = pickle.load(open(filename2, 'rb'))
scaler_dens = pickle.load(open(filename3, 'rb'))
scaler_uncerts = pickle.load(open(filename4, 'rb'))

# FOR TESTING SAMPLE IMAGE
acc = 0
prob = []
for i in range(4900, 5100):
    test_image = x_test[i:i+1]
    test_label = y_test[i:i+1]
    xpred, _, xact = net.predict(test_image)
    uncerts = get_montecarlo_predictions(net, test_image, num_iter=10).var(axis=0).mean(axis=1)
    hid_acts = xact[-2]
    score = loaded_kdes[xpred[0]].score_samples(np.reshape(hid_acts, (1, -1)))[0]

    # Z-score using StandardScaler-models
    uncerts_z = scaler_uncerts.transform([uncerts])
    score_z = scaler_dens.transform([score])

    values = np.concatenate((score_z.reshape((1, -1)), uncerts_z.reshape((1, -1))), axis=0).transpose([1, 0])

    # Predict using LogisticRegressionCV-model
    prob.append(loaded_logreg.predict(values))

    if np.where(np.argmax(test_label,1))[0] < 11:
        true = 0
    else:
        true = 1
    if prob == true:
        acc = acc +1

    #print("dens: {}".format(dens))
    #print("uncert: {}".format(xact1))
    #print("predicted label: {}".format(prob))
    #print("true label: {}".format(test_label))
print("Accuracy: {}".format(acc/400))
# Compute ROC and AUC
n_samples = 100

_, _, auc_score = compute_roc(
    probs_neg=prob[:n_samples],
    probs_pos=prob[n_samples:],
    plot=False
)
print('Detector ROC-AUC score: %0.4f' % auc_score)