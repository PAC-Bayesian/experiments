import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sklearn
from sklearn.cluster import KMeans
from sklearn import metrics


def smooth_hist(count):
    """Moving average 1-d hist counts and return frequencies."""

    # smooth left end and right end 
    left_count, right_count = count[0], count[-1]
    count_extend = np.concatenate([np.array([left_count / 2] * 2), \
    count[1:-1], \
    np.array([right_count / 2] * 2)])
    # moving average neighbors
    count_smooth = np.sum(np.vstack([count_extend[:-2], count_extend[1:-1], count_extend[2:]]), axis=0)
    return count_smooth / count_smooth.sum() 

def calculate_hist(samples_1, samples_2, bins=25, smooth=False):
    """Calculate 1-d hist frequencies of both samples for contrast."""

    lb = np.minimum(samples_1.min(), samples_2.min())
    ub = np.maximum(samples_1.max(), samples_2.max())
    r = (lb, ub) # take union of domains
    count_1, bin_edges_1 = np.histogram(samples_1, range=r, bins = bins)
    count_2, bin_edges_2 = np.histogram(samples_2, range=r, bins = bins)
    assert np.all(bin_edges_1 == bin_edges_2), "not the same bins"
    if smooth:
        return smooth_hist(count_1), smooth_hist(count_2), bin_edges_1
    else:
        return count_1 / count_1.sum(), count_2 / count_2.sum(), bin_edges_1

def calculate_hist_2d(samples_1, samples_2, bins=20):
    r = []
    for i in range(2):
        lb = np.minimum(samples_1[:, i].min(), samples_2[:, i].min())
        ub = np.maximum(samples_1[:, i].max(), samples_2[:, i].max())
        r.append((lb, ub))

    count_1, *_ = np.histogram2d(samples_1[:, 0], samples_1[:, 1], range=r, bins=bins)
    count_2, *_ = np.histogram2d(samples_2[:, 0], samples_2[:, 1], range=r, bins=bins)
    
    return count_1 / count_1.sum(), count_2 / count_2.sum()

def calculate_Hellinger_dist(p, q):
    return np.sqrt(1 - np.sqrt(p * q).sum())

def contrast(collection, bins, label_dict=None, benchmark=None, **kw):
    obj = kw.get('obj', "value_min")
    input_dim = kw.get('input_dim', 1)
    assert benchmark in collection, "no benchmark in the collection is assigned."
    s_0 = collection[benchmark]
    if not label_dict:
        label_dict = {k: k for k in collection}
    num_cols = s_0.shape[-1]
    
    if obj == "value_min":
        jump = 1
    elif obj == "x_argmin":
        jump = input_dim
        
    dist = dict()
    for k in collection:
        if k != benchmark:
            name = label_dict[benchmark] + ' vs ' + label_dict[k]
            dist[name] = []
            s = collection[k]
            for col_0 in range(0, num_cols, jump):
                for col in range(0, num_cols, jump):
                    if obj == "x_argmin" and input_dim == 2:
                        hist = calculate_hist_2d(s_0[:, col_0:col_0+jump], s[:, col:col+jump], bins=bins)
                    else:
                        hist = calculate_hist(s_0[:, col_0:col_0+jump], s[:, col:col+jump], bins=bins)[:-1]
                    dist[name].append(calculate_Hellinger_dist(*hist))
    return dist


### modes matching
def expand_base(*num_comps):
    """Return a mixed-base expansion for decoding.
    
    >>> expand_base(2, 3, 4)
    (12, 4, 1)
    """
    assert num_comps, "zero dimension"
    assert all(type(nc) == int for nc in num_comps), "number of component for each dimension must be int"
    assert all(nc >= 1 for nc in num_comps), "at least 1 component for each dimension"

    bases = num_comps[1:] + (1,)
    return tuple(np.cumprod(np.flip(bases, 0)))[::-1]

def decode(labels, num_comps):
    """Decode data labels (component indices) into integers.

    >>> decode([[0, 1, 2], [1, 2, 3]], (2, 3, 4))
    [6, 23]
    """
    labels = np.array(labels)
    bases = expand_base(*num_comps)
    assert labels.ndim >= 2, "data must be array of 2 dimensions"
    assert labels.shape[-1] == len(num_comps), "label and num_comps must have the same length"
    assert labels.dtype == int, "each component index in label must be int"
    comp_inds = []
    for ci in labels:
        assert np.all(ci <= num_comps), "component index out of range"
        comp_inds.append(ci @ bases)
    return comp_inds

def cluster_density(label, num_cluster):
    label = np.array(label)
    return [sum(label == i) / len(label) for i in range(num_cluster)]

def cluster_kmeans_1d(X, X2, **kw):
    model = KMeans(**kw)
    model.fit(X)
    return model.labels_, model.predict(X2), model







    

