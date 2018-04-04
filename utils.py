import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# TODO: Enable dropout learning_phase(0) in our CNN to enable Monte Carlo search.

def get_montecarlo_predictions(net, x_test, num_iter=50):
    """
    FOR BAYESIAN NEURAL NETWORK UNCERTAINTY.

    :param net: cnn network.
    :param x_test: test images.
    :param num_iter: number of iterations of Monte Carlo sampling.
    :return:
    """

    acts_mc = []
    for i in range(num_iter):
        _, _, activations = net.predict(x_test)  # FIX DROPOUT
        acts_mc.append(activations[-1])

    return np.asarray(acts_mc)


def get_hidden_representations(net, x_test):
    """
    FOR KERNEL DENSITY ESTIMATION.
    Predict given test images and returns last hidden layer activations.

    :param net:
    :param x_test:
    :return:
    """

    _, _, activations = net.predict(x_test)
    activations = activations[-2]

    return np.asarray(activations)


def score_point(tup):
    """
    Helper function for score_samples. Computes KDE scores.

    :param tup: tuple of samples and kernel density.
    :return: kernel density score of samples.
    """

    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kernel_dens, samples, predictions, n_jobs=None):
    """
    Computes the kernel density score for each sample for predicted class.

    :param kernel_dens:
    :param samples:
    :param predictions:
    :param n_jobs: number of processes allowed to compute in parallel.
    :return:
    """

    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kernel_dens[i]) for x, i in zip(samples, predictions)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, novelty):
    """
    Normalize data.

    :param normal:
    :param novelty:
    :return:
    """

    n_samples = len(normal)
    total = scale(np.concatenate((normal, novelty)))

    return total[:n_samples], total[n_samples:]


def train_logistic_regression(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    Train the logistic regression model given densities and uncertanties.

    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """

    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr


def compute_roc(probs_neg, probs_pos, plot=False):
    """
    Compute ROC and AUC score.

    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """

    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score