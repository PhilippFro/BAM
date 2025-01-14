import numpy as np
import tensorflow as tf
from tensorflow import keras
import math

class DataGeneratorChebyshev(keras.utils.Sequence):
    """Generator for graph and data, where data dependencies are
        Chebyshev polinomials with randomly chosen coefficients and
        factorial decrease
    """
    def __init__(self, N, M_min, M_max, d_min, d_max, steps_per_epoch=128):
        self.N = N
        self.M_min = M_min
        self.M_max = M_max
        self.d_min = d_min
        self.d_max = d_max
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        self.data = self.__data_generation()
        return self.data

    def __data_generation(self):
        d = np.random.randint(self.d_min, self.d_max + 1)
        expected_degree = tf.cast(np.random.randint(size=(1,), low=1, high=np.min((np.ceil(d / 3).astype(int) + 1,5))), tf.int32)[0]
        M = np.random.randint(self.M_min, self.M_max + 1)
        def generate_1_sample(i):
            'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
            adjacency=sample_adjacency_expected_parents_ER(d,expected_degree)
            X=sample_from_dag_chebyshev(adjacency,M,100) # at least 100 samples in calculations for stable BatchNorm
            L=adjacency
            np.fill_diagonal(L,1)
            prec = np.matmul(np.transpose(L),L)
            prec = np.array(np.array(prec, dtype=bool), dtype=np.int32)
            causal_prec = adjacency+np.transpose(adjacency)
            np.fill_diagonal(causal_prec,1)
            moralizations = prec - causal_prec
            np.fill_diagonal(moralizations,1)

            no_edge = 1 - prec
            np.fill_diagonal(no_edge,1)

            perm = np.random.permutation(d)
            no_edge = np.take(np.take(no_edge, perm, axis=0), perm, axis=1)
            causal_prec = np.take(np.take(causal_prec, perm, axis=0), perm, axis=1)
            moralizations = np.take(np.take(moralizations, perm, axis=0), perm, axis=1)

            return np.take(X, perm, axis=1), np.stack([no_edge,causal_prec,moralizations],axis=2)

        samples = tf.map_fn(generate_1_sample, tf.range(self.N), dtype=(tf.float32, tf.int32))
        return samples


def batchNorm(activations):
    mean = np.mean(activations, axis=0, keepdims=True)
    variance = np.var(activations, axis=0, keepdims=True)
    activations = (activations - mean) / np.sqrt(variance + 1e-5)
    return activations


def get_scaling(percentiles, threshold=5):
    p2=percentiles.copy()
    p2[2,:]=0
    idx = np.argmax(np.array(p2 <= threshold,float), axis=0)
    sc = np.zeros(idx.shape)
    sc[idx == 0] = percentiles[0, idx == 0]
    sc[idx == 1] = percentiles[1, idx == 1]
    sc[idx == 2] = percentiles[2, idx == 2]
    return sc


def truncatedChebyMultidim(X, c, c_multidim, threshold=5):
    X = np.array(X)
    X=X-np.mean(X,0,keepdims=True)
    percentiles = np.percentile(np.abs(X), (100, 95, 90), axis=0)
    sc=get_scaling(percentiles, threshold=5)
    sc=np.maximum(sc,1e-5)
    result = np.polynomial.chebyshev.chebval(X / np.expand_dims(sc,0), c)
    result = scale_and_threshold(result,threshold)
    result2=0
    if np.abs(c_multidim)>0.001:
        X_eps = np.column_stack((scale_x(X, sc), np.random.uniform(-0.5, 0.5, X.shape[0])))
        shift_x=np.random.normal(0,0.5,1)
        shift_y=np.random.normal(0,0.5,1)

        mixed_terms = (X_eps[:, :, np.newaxis]-shift_x) * (X_eps[:, np.newaxis, :]-shift_y)/(1+np.abs(shift_x))/(1+np.abs(shift_y))
        lower_diag_indices = np.tril_indices(X_eps.shape[1], k=-1)
        mixed_terms = mixed_terms[:, lower_diag_indices[0], lower_diag_indices[1]]
        mixed_terms_s=scale_and_threshold(mixed_terms,threshold)

        coef_mixed=np.random.uniform(-1,1,mixed_terms_s.shape[1])
        coef_mixed=coef_mixed/np.sum(np.abs(coef_mixed))*c_multidim
        result2=np.matmul(mixed_terms_s,coef_mixed)
    res_additive=abs_average(result,weights=np.random.uniform(0.7,1.3,X.shape[1])/X.shape[1],axis=1)
    result3=result2+res_additive
    result3=scale_and_threshold(result3,threshold)
    part_eps=np.random.uniform(0.1,0.4,1)
    eps=scale(sample_GaussianMixture(X.shape[0]))
    new_attribute=scale(part_eps*eps+(1-part_eps)*result3)
    return new_attribute


def sample_c2(degree_Cheb=5):
    # Using statistically decreasing coefficients as in Trefethen (2008)
    num_non_zero = np.random.randint(1, degree_Cheb)
    w = np.random.choice(degree_Cheb - 1, num_non_zero, False) + 1
    c = np.zeros(degree_Cheb)
    c[w] = np.random.uniform(-1, 1, num_non_zero)
    c_multidim=np.random.uniform(-1, 1,1)*np.random.binomial(1,0.5,1)
    c=c/[math.factorial(i) for i in range(0, degree_Cheb)]
    s=np.sum(np.abs(c))+np.abs(c_multidim)
    c = c/s
    c_multidim=c_multidim/s
    return c, c_multidim


def sample_from_dag_chebyshev(A_dag, N, N2=200):
    d = A_dag.shape[0]
    X = np.zeros((np.max((N2,N)), d))
    for l in range(d):
        pa = np.where(A_dag[l,:] == 1)[0]
        c, c_multidim = sample_c2()

        if len(pa) != 0:
            x = truncatedChebyMultidim(X[:,pa], c, c_multidim)
            #x=random_Cheby_multidim(X[:,pa])
            X[:,l] = x
        else:
            x=sample_GaussianMixture(max(N,N2))
            x = (x - x.mean()) / x.std()
            X[:,l] = x
    if N<N2:
        X=X[0:N, :]
        X=batchNorm(X)
    return X


def sample_adjacency_expected_parents_ER(d, expected_degree):
    p=expected_degree/(d-1)
    A = np.random.binomial(1,p,(d,d))
    np.fill_diagonal(A, 0)
    A[np.triu_indices(n=d, k=1)] = 0
    return A

def sample_upper_diagonal_matrix_np(d, p):
    samples = np.random.normal(size=(d * (d - 1) // 2,)) * np.random.binomial(n=1, p=p, size=(d * (d - 1) // 2,))
    L = np.zeros((d, d))
    L[np.triu_indices(d, k=1)] = samples
    np.fill_diagonal(L, 1)
    del samples
    samples = None
    return L


def scale(x):
    x = (x - np.mean(x)) / np.maximum(np.std(x),0.0001)
    return x


def sample_GaussianMixture(sampleSize,maxNumComponents=5):
    num_components = np.random.randint(1, maxNumComponents)
    means = np.random.uniform(-1, 1, size=num_components)
    sds = np.random.uniform(0.05, 1, size=num_components)
    weights = np.random.uniform(0.3, 1, size=num_components)
    weights /= weights.sum()
    component_choices = np.random.choice(num_components, size=sampleSize, p=weights)
    x = np.array([np.random.normal(means[i], sds[i]) for i in component_choices])
    return x

def scale_x(X,sc):
    Xsc=X/sc
    Xsc=np.maximum(Xsc,-1)
    Xsc=np.minimum(Xsc,1)
    return Xsc

def scale_and_threshold(result,threshold):
    result = batchNorm(result)
    sc_tanh = np.maximum(1,(np.max(np.abs(result))-threshold)/2)
    result[result < -threshold] = -threshold - np.tanh((-result[result < -threshold]-threshold)/sc_tanh)
    result[result > threshold] = threshold + np.tanh((result[result > threshold]-threshold)/sc_tanh)
    result = batchNorm(result)
    return result

def abs_average(X,weights,axis):
    weighted_sum = np.sum(X * weights,axis=axis)
    return weighted_sum