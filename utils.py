import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import scale, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegressionCV
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def get_montecarlo_predictions(net, x_test, num_iter=50):
    """
    Does a Monte Carlo search of activation variations in the logits to find uncertainties.

    :param net: cnn network.
    :param x_test: test images.
    :param num_iter: number of iterations of Monte Carlo sampling.
    :return:  array of logits from each iteration.
    """

    acts_mc = []
    for i in range(num_iter):
        _, _, activations = net.predict(x_test, dropout_enabled=True)
        acts_mc.append(activations[-1])

    return np.asarray(acts_mc)


def get_hidden_representations(net, x_test):
    """
    Predict given test images and returns last hidden layer activations.

    :param net: cnn network.
    :param x_test:
    :return: array or activations.
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
    :param samples: images to be scored.
    :param predictions: predicted label of sample.
    :param n_jobs: number of processes allowed to compute in parallel.
    :return: array of scores.
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

    :param normal: closed set images (images belonging to training set domain).
    :param novelty: open set images (novelties).
    :return: normalized data and scale models.
    """

    n_samples = len(normal)

    values = np.hstack((normal,novelty)).reshape(-1,1)

    total_orig = scale(np.concatenate((normal, novelty)))

    scaler = StandardScaler()
    total = scaler.fit_transform(values)

    scaler2 = RobustScaler()
    total2 = scaler2.fit_transform(values)

    return total[:n_samples].ravel(), total[n_samples:].ravel(), scaler, \
           total2[:n_samples].ravel(), total2[n_samples:].ravel(), scaler2


def train_logistic_regression(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    Train the logistic regression model given densities and uncertainties.

    :param densities_pos: densities for open set images (novelties).
    :param densities_neg: densities for closed set images (normals).
    :param uncerts_pos: uncertainties for open set images.
    :param uncerts_neg: uncertainties for closed set images.
    :return: all samples and labels plus the logistic regression model.
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

    :param probs_neg: predicted labels for closed set images.
    :param probs_pos: predicted labels for open set images.
    :param plot: boolean for plotting or not.
    :return: false positives, true positives, area-under-curve score.
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
