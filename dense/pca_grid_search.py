# import svm
# import scatnet_utilities

import numpy as np
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
import argparse
import pandas as pd

sys.path.append("/home/lermang/vonse006/ScatteringTransform/wavelet-texture-synthesis/")
# sys.path.append('/home/lermang/vonse006/ScatteringTransform/directional/')
# from routine import call_lbfgs2_routine
# from load_image import load_image_gray
# from hist import *
from ops.alpha_gray import ALPHA

from sklearn import preprocessing
from sklearn import svm as svm
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA


class PCAClassifier(BaseEstimator):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        # Validate input data
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Store models
        self.pcas_ = {}
        self.scalers_ = {}


        # Fit PCA and scaler for each class
        for class_label in self.classes_:
            X_class = X[y == class_label]
            scaler = preprocessing.StandardScaler()
            X_class_scaled = scaler.fit_transform(X_class)
            pca = PCA(n_components=self.n_components)
            pca.fit(X_class_scaled)
            self.pcas_[class_label] = pca
            self.scalers_[class_label] = scaler

        return self

    def predict(self, X):
        # Check if the estimator is fitted
        check_is_fitted(self)

        # Convert input to 2D array
        X = check_array(X)

        dists = []
        for label in self.classes_:
            pca_tmp = self.pcas_[label]
            scaler_tmp = self.scalers_[label]
            X_scaled = scaler_tmp.transform(X)
            dists_tmp = dist_to_subspace(X_scaled, pca_tmp.components_.T)
            dists.append(dists_tmp)

        dists = np.stack(dists)
        argmins = np.argmin(dists, axis=0)
        return self.classes_[argmins]

def dist_to_subspace(pts, basis):
    num_sample = pts.shape[0]
    proj = project_onto_vectors(pts=pts, basis=basis)
    dists = proj - pts
    dists = np.linalg.norm(dists, axis = 1)
    return(dists)

def project_onto_vectors(pts, basis, weights = None):
    """
    Project a set of points onto a collection of vectors.

    Args:
        pts (np.ndarray): Matrix of shape [num_samples, Dim] representing the points.
        basis (np.ndarray): Matrix of shape [Dim, num_basis_vecs] representing the basis vectors.

    Returns:
        np.ndarray: Matrix of shape [num_samples, num_basis_vecs] representing the projections.
    """
    # Normalize the basis vectors
    basis_norms = np.linalg.norm(basis, axis=0)
    basis_norms[basis_norms == 0] = 1  # Handle zero norms to avoid division by zero
    basis_normed = basis / basis_norms
    
    if weights == None:
        # Calculate the weights of the projections
        weights = np.dot(pts, basis_normed)

    # Calculate the projections
    projections = np.dot(weights, basis_normed.T)

    return (projections)



def cv_grid_search_svm(J, L, A,  train_loader, num_samples = 1000, normalize = True,
                      param_grid = {'n_components':[10,20,50]}):
    M = N = args.resize
    A_prime = args.A_prime
    dj = args.dj
    nb_chk = 1
    chk_id = 0
    shift = args.shift
    wavelets = args.wavelets

    hatpsi = torch.load(
        "/home/lermang/vonse006/ScatteringTransform/wavelet-texture-synthesis/image-classification/filters/morlet_N"
        + str(N)
        + "_J"
        + str(J)
        + "_L"
        + str(L)
        + ".pt"
    )  # (J,L,M,N)
    hatphi = torch.load(
        "/home/lermang/vonse006/ScatteringTransform/wavelet-texture-synthesis/image-classification/filters/morlet_lp_N"
        + str(N)
        + "_J"
        + str(J)
        + "_L"
        + str(L)
        + ".pt"
    )  # (M,N)
    filters = {"hatpsi": hatpsi, "hatphi": hatphi}

    phase_harm = ALPHA(
        M, N, J, L, A, A_prime, dj, nb_chk, chk_id, shift, wavelets, filters=filters
    )
    
    feats = []
    labels = []
    for idx, (data, label) in enumerate(train_loader):
        print(idx)
        if idx * args.batch_size < num_samples:
            feats.append(phase_harm(data))
            labels.append(label)
        else:
            break
            
    feats = torch.concat(feats).numpy()
    labels = torch.concat(labels).numpy()
    
    
    gridSearch = GridSearchCV(estimator=PCAClassifier(), param_grid = param_grid, scoring = 'balanced_accuracy', n_jobs=-2)
    gridSearch.fit(feats, labels)
    
    results = pd.DataFrame(gridSearch.cv_results_)
    results['J'] = J
    results['A'] = A
    results['L'] = L
    results['num_samples'] = num_samples
    
    return(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--N', type=int, default=256)
    # parser.add_argument('--image', default='gravel')
    # parser.add_argument('--model', default='alpha')
    parser.add_argument('--J', type=int, default=3)
    parser.add_argument('--L', type=int, default=4)
    parser.add_argument("--dj", type=int, default=4)
    parser.add_argument('--A', type=int, default=4)
    parser.add_argument("--A_prime", type=int, default=1)
    parser.add_argument("--wavelets", default="morlet")
    parser.add_argument("--shift", default="samec")  # 'samec' or 'all'
    # parser.add_argument('--nb_chunks', type=int, default=11)
    # parser.add_argument('--nb_restarts', type=int, default=3)
    # parser.add_argument('--nGPU', type=int, default=1)
    # parser.add_argument('--maxite', type=int, default=500)
    # parser.add_argument('--factr', type=int, default=1e-3)
    # parser.add_argument('--nb_syn', type=int, default=3)
    # parser.add_argument('--hist', action='store_false')
    # parser.add_argument('--save', action='store_true')
    # parser.add_argument('--plot', action='store_false')
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--resize", default=32, type=int)
    parser.add_argument("--num-samples", default = 1000, type = int)
    
    args = parser.parse_args()

    mnist_dir = "/home/lermang/vonse006/data/"

    mnist_train = torchvision.datasets.MNIST(
        mnist_dir,
        download=True,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs,
                transforms.Resize(args.resize),
                transforms.Grayscale(num_output_channels=3),
            ]
        ),
    )
    mnist_test = torchvision.datasets.MNIST(
        mnist_dir,
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),  # first, convert image to PyTorch tensor
                transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
                transforms.Resize(args.resize),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    results = []

    for J in [2,3,4,5]:
        for L in [4,8,12]:
            for A in [4,8,12]:
                print(J, L, A)
                result_tmp = cv_grid_search_svm(J = J, L = L, A = A, train_loader=train_loader, num_samples=args.num_samples, param_grid = {'n_components': np.arange(10, int(args.num_samples/10*0.8), 10)[:-1]})
                results.append(result_tmp)
                result_tmp.to_csv(f'intermediate_{args.num_samples}_samples_crossval_PCA.csv', mode = 'a')

    all_results = pd.concat(results)
    all_results.to_csv(f'{args.num_samples}_samples_crossval_PCA.csv')