import numpy as np
from scipy.stats import entropy
import torch
from tqdm import tqdm
from collections import Counter

class KLDivergence:
    def __init__(self, eps=1e-5, laplacian_smoothing = False, mode='normal', size=101, precision=np.int16):
        '''
        eps: Add epsilon to p, q arrays to avoid divsion by 0
        laplacian_smoothing: Add 1 to p, q, arrays to perform Laplacian Smoothing
        mode: Varies between 'normal', 'both', and 'symmetric'.
        'normal' returns KLD from p to q.
        'both' returns KLD from p to q and q to p in form of a tuple
        'symmetric' returns mean(KLdist(p->q), KLdist(q->p))
        size: Size of the distribution
        '''
        self.laplacian_smoothing = laplacian_smoothing
        if self.laplacian_smoothing:
            self.eps = 1.
        else:
            self.eps = eps
        self.mode = mode
        self.fitted = False
        self.size = size
        self.label_0_dist = np.zeros(self.size)
        self.label_1_dist = np.zeros(self.size)
        self.label_0_mean_kld = None
        self.label_1_mean_kld = None
        self.precision = precision
        
    def _distance(self, p, q):
        '''
        Instance method to calculate KL-Divergence. 
        p: First distribution of type np.ndarray or list
        q: Second distribution of type np.ndarray or list
        
        Returns:
        KL-Divergence from p to q using scipy.entropy backend. Returns distance according to self.mode.
        '''
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)

        p, q = p + self.eps, q + self.eps # avoid 0s
        if self.mode == 'both':
            return (entropy(p, qk=q), entropy(q, qk=p))
        if self.mode == 'symmetric':
            return (entropy(p, qk=q) + entropy(q, qk=p)) / 2.0

        return entropy(p, qk=q) # pairwise Kullback-Leibler Divergence for mode='normal'
    
    def dist(self, p, q, indexify_p=True, indexify_q=True):
        '''
        Wrapper instance method for self._distance
        '''
        len_p, len_q = len(p), len(q)
        #assert len_p == len_q, "Distribution length mismatch! Dist 1 length was {} while Dist 2 was {}".format(len_q, len_q)
        assert isinstance(p, (list, np.ndarray)), "Dist 1 must be a list or numpy array, found type {}".format(type(p))
        assert isinstance(q, (list, np.ndarray)), "Dist 2 must be a list or numpy array, found type {}".format(type(q))

        if indexify_p and indexify_q:
            return self._distance(self._indexify(p), self._indexify(q))
        
        if indexify_p:
            return self._distance(self._indexify(p), q)
        
        if indexify_q:
            return self._distance(p, self._indexify(q))
    
        return self._distance(p, q)
    
        #return self._distance(p_avg_arr, q_avg_arr)

    def _indexify(self, indices_arr):
        '''
        Expects an array of indices, converts them another array where each index stores the indices seen in the original `indices_arr`. Averages the values.
        Backbone function for the entire module.
        '''
        len_arr = len(indices_arr)
        arr_counts = Counter(indices_arr)
        arr_avg_counts = {index: count / len_arr for index, count in arr_counts.items()}
        num_bins = self.size
        arr_avg_arr = np.zeros(num_bins)

        for idx, count in arr_avg_counts.items():
            arr_avg_arr[idx] = count
        
        return arr_avg_arr



    def fit(self, X, y, return_gof = True):
        '''
        X: Input (n, self.size) distribution where n denotes the number of samples.
        y: Input binary labels (0 and 1) of size (n,).
        precision: Precision for X, y to be formed into numpy arrays.
        return_gof: Whether to return goodness of fit. GOF is the average KLD from the mean distribution formed for each sample.
        
        RETURNS:
        Mean label distribution for 0 and 1, and optionally, goodness of fit KLD value for each label if return_gof = True.
        '''
        assert not self.fitted, "Instance was already fit! Cannot fit more than once."
        assert isinstance(X, (torch.Tensor, np.ndarray, list))
        assert isinstance(y, (torch.Tensor, np.ndarray, list))

        X = np.asarray(X, dtype=self.precision)
        y = np.asarray(y, dtype=self.precision)

        # dimension checks
        assert len(X.shape) == 2 and X.shape[1] == self.size, f"Invalid input shape for X. Expected shape (n, {self.size}) but got {X.shape}!"
        assert len(y.shape) == 1, f"Invalid dimensionality for y. Expected 1 dimension but got {len(y.shape)}!"
        assert len(set(y.tolist())) == 2, f"Invalid number of unique labels. Expected 2 unique labels (0 and 1) but got {len(set(y.tolist()))} unique labels!"

        label_0_dist = self.label_0_dist # initialize label distributions
        label_1_dist = self.label_1_dist
        X_label_0 = X[y == 0] # separate data based on labels
        X_label_1 = X[y == 1]
        
        X_label_0 = X_label_0.flatten()
        X_label_1 = X_label_1.flatten()

        label_0_dist = self._indexify(X_label_0)
        label_1_dist = self._indexify(X_label_1)

        self.label_0_dist = label_0_dist # update instance label distributions
        self.label_1_dist = label_1_dist

        self.fitted = True

        if not return_gof:
            return {'label_0': self.label_0_dist, 'label_1': self.label_1_dist}
        
        # Estimate goodness of fit
        print("Estimating goodness of fit...")
        label_0_kld, label_1_kld = 0.0, 0.0
        label_0_c, label_1_c = 0+self.eps, 0+self.eps # avoid division by zero
        for idx, sample in tqdm(enumerate(X)):
            if y[idx] == 0:
                label_0_kld += self.dist(sample, self.label_0_dist, indexify_p=True, indexify_q=False)
                label_0_c += 1
            else:
                label_1_kld += self.dist(sample, self.label_1_dist, indexify_p=True, indexify_q=False)
                label_1_c += 1
        self.label_0_mean_kld = label_0_kld / label_0_c
        self.label_1_mean_kld = label_1_kld / label_1_c

        if label_0_c == 0+self.eps:
            print("WARNING: No sample for label 0 was found. KLD estimate is not defined.")
            self.label_0_mean_kld = None
        if label_1_c == 0+self.eps:
            print("WARNING: No sample for label 1 was found. KLD estimate is not defined.")
            self.label_1_mean_kld = None

        return {'label_0': self.label_0_dist, 'label_1': self.label_1_dist,
                'label_0_mean_kld': self.label_0_mean_kld, 'label_1_mean_kld': self.label_1_mean_kld}
        
    def predict(self, X):
        '''
        Returns predictions for the binary label based on closest KLD representative distribution.

        X: Input (n, self.size) distribution where n denotes the number of samples.
        Returns:
        y_pred: (n,) sized list containing binary label predictions based on KLD
        '''
        assert self.fitted, "KLD instance must be fitted before running predictions!"
        X = np.asarray(X, dtype=self.precision)
        if len(X.shape) == 1:
            orig_shape = X.shape
            X = X.reshape((1, X.shape[0]))
        assert len(X.shape) == 2 and X.shape[1] == self.size, f"Invalid input shape for X. Expected shape (n, {self.size}) but got {orig_shape}!"
        y_pred = [None] * X.shape[0]
        for idx, arr in tqdm(enumerate(X)):
            #dist0 = self._distance(arr, self.label_0_mean_kld)
            #dist1 = self._distance(arr, self.label_1_mean_kld)
            
            dist0 = self.dist(arr, self.label_0_dist, indexify_p=True, indexify_q=False)
            dist1 = self.dist(arr, self.label_1_dist, indexify_p=True, indexify_q=False)
            label = 0 if dist0 < dist1 else 1
            y_pred[idx] = label
        
        return np.array(y_pred)


    def __repr__(self):
        s = '''
        This is a KL-Divergence measurement module by GM Harshvardhan. Uses scipy.entropy as backend. Look at KLDivergence_legacy for manual implementation.
        ***********************************************************************
        Parameters of object:
        Epsilon: {}
        Laplacian smoothing enabled: {}
        Mode: {}
        Fitted?: {}
        Distribution size: {}
        Precision: {}
        '''.format(self.eps, self.laplacian_smoothing, self.mode, self.fitted, self.size, self.precision)
        return s



class KLDivergence_legacy:
    np.seterr(invalid='ignore') # if same distribution is passed, ignore 0/0 warning
    def __init__(self, eps=1e-5):
        self.eps = eps
    
    def _distance(self, p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        
        # Check if p or q do not sum to 1, normalize if not
        if p.sum() != 1 or q.sum() != 1:
            p /= p.sum()
            q /= q.sum()
        
        p, q = p + self.eps, q+self.eps # avoid 0s

        #return np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return np.sum(p * np.log(p/q))
    
    def dist(self, p, q):
        len_p, len_q = len(p), len(q)
        assert len_p == len_q, "AssertionError: Distribution length mismatch! Dist 1 length was {} while Dist 2 was {}".format(len_q, len_q)
        assert isinstance(p, (list, np.ndarray)), "AssertionError: Dist 1 must be a list or numpy array, found type {}".format(type(p))
        assert isinstance(q, (list, np.ndarray)), "AssertionError: Dist 2 must be a list or numpy array, found type {}".format(type(q))
        return self._distance(p, q)
    





