"""TensorFlow-free copy of the original BAM Chebyshev simulation helpers.

The numerical helper bodies are extracted unchanged from generator_cheby_BAM.py
on the archived BAM_old branch (upstream commit d8319340dc91a13d46eebf8f1e6e447a182964e5).
They preserve its child,parent adjacency convention, topologically ordered input,
NumPy-global RNG semantics, coefficient distribution, and noise construction.
Use joint_data_BAM.sample_batch for source,target labels and isolated RNG state.

Original work: Philipp Froehlich, https://github.com/PhilippFro/BAM
Copyright (c) 2025 Philipp Froehlich. Distributed under the MIT License; see LICENSE.
"""

import math
import numpy as np


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
