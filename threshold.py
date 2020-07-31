import random
from sklearn.neighbors import KernelDensity
import math
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from matplotlib import pyplot as plt
import copy

np.set_printoptions(suppress=True)  # scientific notation off
pi = math.pi


def _gen_correlated_data(n, seed=5, df_type='int'):
    """
    Generate data containing signal and noise, run correlation statistics on that data

    Parameters
    -----------
    n : int
        dimension of the n x n matrix

    seed :
        numpy random seed

    df_type:
        int or float entries
    Return
    -------
    type : pd.DataFrame
        correlation matrix

    """
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(low=0, high=1024, size=(n, n))
    for i in range(len(arr)):
        for j in range(len(arr)):
            if j % 7 == 0:         # add signals here: the greater the module (less signals), the higher the threshold
                arr[i][j] = i       # generate perfect correlations (1.0)

    arr = np.asarray(arr)
    # arr = normalize_entries(arr, 'no')  # 'yes' or 'no' for mean 0, variance 1 of initial data

    idx_labels = ['row_{}'.format(i) for i in range(n)]
    col_labels = ['col_{}'.format(i) for i in range(n)]

    if df_type == 'int':
        random_df = pd.DataFrame(np.asarray(arr),
                                 index=idx_labels,
                                 columns=col_labels)
    elif df_type == 'float':
        random_df = pd.DataFrame(np.asarray(arr),
                                 index=idx_labels,
                                 columns=col_labels)
    else:
        raise ValueError('unexpected df type')

    print("amplicon data")
    print(random_df)                                # print generated data

    return random_df.corr(method='pearson')         # generate correlation matrix from data


def normalize_entries(corr_df, answer='yes'):
    """
    Normalize correlation matrix to have 0 mean and standard deviation 1

    Parameters
    -----------
    corr_df : pd.DataFrame
        initial correlation matrix

    answer : string
        "yes" or "no" for 0 mean and unit variance

    Return
    -------
    type :
        if yes: ndarray (numpy array)
            normalized correlation matrix with 0 mean and unit variance
        if no: pd.DataFrame
            initial correlation matrix
    """

    x = np.asarray(corr_df)
    print("mean:", np.round(x.mean(), 4), "variance:", np.round(x.std(), 4))     # initial mean and variance
    # pos = x[x > 0]
    # neg = x[x < 0]
    # print("positive entries:", pos.size, ", negative entries: ", neg.size)

    if answer == "yes":
        if x.std() != 1.0 and x.mean() != 0.0:
            x = np.round((x - x.mean()) / x.std(), 4)
        elif x.std() != 1.0:
            x = np.round(x / x.std(), 4)
        elif x.mean() != 0.0:
            x = np.round(x - x.mean(), 4)
        else:
            print("entries are already normalized")
            return corr_df

        print("Normalized")
        print("mean:", np.round(x.mean(), 4), "variance:", np.round(x.std(), 4))
        print(x)
        return x

    elif answer == "no":
        return corr_df
    else:
        print("Please enter yes or no.")


def plot(nnsd, n):
    """
    Plot final NNSD after it diverges from the Poisson

    Parameters
    -----------
    nnsd : ndarray
        nearest neighbor spacial distribution points

    n : int
        number of points and bins to histogram the nnsd density

    Return
    -------
    plot: NNSD

    NOTE: Not necessary to determine the threshold value, but helpful sanity check
    """

    x = np.linspace(0, 10)          # domain of graph

    # Wigner-Dyson Distribution:
    goe = [(x[i] * pi / 2) * math.exp((-1 * pi * x[i] ** 2) / 4) for i in range(0, len(x))]

    # Negative Exponential Distribution:
    poisson = [math.exp(-x[i]) for i in range(0, len(x))]

    # nearest neighbor spacial distribution
    # plt.subplot(1, 2, 2)
    plt.plot(x, goe, 'b', label="GOE")
    plt.plot(x, poisson, 'g', label="Poisson")
    plt.title("NNSD")
    plt.ylabel("Density")
    plt.hist(nnsd, bins=n, density=True)
    plt.xlabel("Eigenvalues")
    plt.axis([-0.1, x, 0, 1.0])  # window
    plt.legend(loc="upper right")
    plt.show()


def make_symmetric(matrix):
    # M = ½(A + AT):
    # answer = input("Make correlation matrix symmetric? (y/n): ")
    answer = "y"
    if answer == "y":
        trans = np.transpose(matrix)
        h = np.add(matrix, trans) / 2  # make symmetric
        return h
    else:
        return matrix


def _compute_threshold_w_random_theory(corr_df):
    """
    Data pre-processing
        0. normalize correlation matrix
        1. Establish an initial candidate threshold
        2. send information to calculate_threshold_value

    Data post-processing
        Scan for signals based on calculated threshold value

    Parameters
    -----------
    corr_df: pd.DataFrame
        initial correlation matrix

    Return
    -------
    type : float
        final threshold value
    """
    # step (0): Given correlation matrix A  (see app[0] at the bottom)
    print(corr_df)
    norm_matrix = normalize_entries(corr_df)            # normalize correlation matrix 0 mean 1 variance
    arr = np.asarray(norm_matrix)                       # normalized corr. matrix array
    corr_matrix = np.asarray(corr_df)                   # original corr. matrix array

    # step (1): Establish an initial candidate threshold: t = max(|a[i][j]|) - alpha (0.01 usually)
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 0)                           # ignore diagonals while finding max (they are 1.0)
    max_value = np.absolute(corr_matrix[mask]).max()    # t = candidate threshold
    t = max_value
    if t < 0.2:                                         # if t small, start the thresh. decrement smaller also
        threshold_decrement = 0.01
    else:
        threshold_decrement = 0.1

    #  threshold decrement goes 0.1 --> 0.01 --> 0.001
    threshold = calculate_threshold_value(arr, t, threshold_decrement)
    print("Final Threshold Value: ", threshold)

    # Begin scan for signals based on threshold value:

    x = np.asarray(corr_df)
    values = np.absolute(x)  # take absolute values of elements

    # only look at values above/below the diagonal entries, the matrix is symmetric and we don't want to double count
    # for mutual information, we would want to look at BOTH upper and lower triangles of matrix, just not diag.
    lower_tri = values[np.triu_indices(1000, k=1)]       # k = 1 for symmetry about the diagonal

    signals = lower_tri[lower_tri >= threshold]
    sig_percent = (signals.size / lower_tri.size) * 100  # percent of signals in data
    print("Potential Signals Detected: ", signals.size, ", %{}".format(sig_percent))

    # print(signals)                                     # print values of signals
    # x = np.where(lower_tri >= threshold)
    # print("with positions: ", "\n", np.asarray(x).T)   # print positions of signals


def calculate_threshold_value(arr, thresh, thresh_dec):
    """
    computes threshold value based on random matrix theory for a correlation matrix
        2. Construct a new network, A' based on threshold value
        3. Calculate eigenvalues of A', sort in increasing order
        4. Apply spectral unfolding technique to correct heterogeneity
        5. calculate NNSD
        6. Run chi-squared test to determine similarity of NNSD to GOE
        7. Decrease threshold value by threshold decrement until desired result is achieved
    Parameters
    -----------
    arr: ndarray
        normalized correlation matrix array

    thresh: float
        initial threshold candidate

    thresh_dec: float
        how much we decrease the threshold value by to begin with

    Return
    -------
    type : float
        final threshold value
        
    Note: optionally can send information to be plotted here
    """

    original = copy.deepcopy(arr)                       # deep copy of original arr

    result = 50                                         # default to less than 100
    while result < 100:                                 # while result of chi-squared is more than ___ (usually 100)
        arr = copy.deepcopy(original)
        thresh -= thresh_dec
        if math.isclose(thresh, 0) or thresh < 0:       # if the threshold value is close to 0, raise error
            raise Exception("Data might be all Noise")

        # step (2)
        # Construct a new network, A', such that:
        #   a[i][j] = 0  if  t - |a[i][j]| > 0
        #   a[i][j] = a[i][j] if  t - |a[i][j]| <= 0

        for i in range(0, len(arr)):
            for j in range(0, len(arr)):
                if i != j:  # if not a diagonal entry
                    if thresh - abs(arr[i][j]) > 0:
                        arr[i][j] = 0

        # print(f"frequency {count / (len(arr)**2 - len(arr))}")
        # print(arr)

        # step (3)
        # Calculate eigenvalues of A'
        eigen_value = np.linalg.eigvalsh(arr)
        eigen_value.sort()

        # step (4)
        # (i) Apply spectral unfolding technique to correct heterogeneity (see app[1])
        kde = KernelDensity(bandwidth=0.1, kernel='gaussian')  # see app[2] for bandwidth
        kde.fit(eigen_value[:, None])
        eigen_value = kde.score_samples(eigen_value[:, None])
        eigen_value = [abs(eig) for eig in eigen_value]
        eigen_value = np.sort(eigen_value)

        # (ii) Calculate the NNSD (nearest neighbor spacial distribution, P(X))
        eigen_spacing = [eigen_value[i + 1] - eigen_value[i] for i in range(0, len(eigen_value) - 1)]
        # print(eigen_spacing)
        average_spacing = sum([abs(eig) for eig in eigen_spacing]) / (len(eigen_spacing))  # returns a float
        if math.isclose(average_spacing, 0):
            raise Exception("Error in Data")  # occurs if threshold too high, diagonals all 1, other entries 0
        nnsd = ([eig / average_spacing for eig in eigen_spacing])
        # print("average spacing:", average_spacing)

        n = 60  # number of bins to group nnsd.
        # NOTE: More bins, greater threshold value, less bins, smaller thresh
        # n = 60 seems good, however, needs further investigation.

        # (distribution of NNSD, P(X))
        y_values, x_values, _ = plt.hist(nnsd, bins=n, density=True)  # get x and y values of histogram
        plt.clf()

        domain = x_values.max()
        # we restrict the domain to counter-act extreme outlier points when comparing distributions
        # a better method entirely would be a function which scanned for and discarded such outliers
        if domain > 20:  # this is a parameter that needs to be optimized/explored further
            domain = 20  # 20 seemed to work well for our purposes

        x = np.arange(0., domain, domain / n)              # equidistant n number of x values on the domain of the NNSD
        neg_exp = [math.exp(-x[i]) for i in range(0, len(x))]  # Poisson distribution data points
        if len(neg_exp) > len(y_values):                   # had some issues with uneven arrays, this pops last element
            neg_exp.remove(neg_exp[-1])
        result, _ = scipy.stats.chisquare(y_values, f_exp=neg_exp, ddof=int(n-1), axis=0)  # goodness-of-fit test

        # Does P(x) follow the negative exponential distribution? see app[3]
        #   if yes: recalculate candidate threshold, t = t - dec., go to step (2)
        #   if no: return threshold value

        print("Chi-squared statistic: ", result, "Threshold value: ", thresh)

    if thresh_dec > 0.0001:  # check 0.1, 0.01, 0.001 places to determine precise threshold value
        return calculate_threshold_value(original, thresh + thresh_dec, thresh_dec / 10)
    else:
        # plot(nnsd, n)      # plot function for sanity check: make sure it has diverged and somewhat resembles GOE
        return thresh


# ------------------------------------------------------------------------------------------------------------------- #


# run and test random matrix theory
random_theory_threshold = _compute_threshold_w_random_theory(_gen_correlated_data(1000))  # parameter is len of matrix
expected_random_theory_threshold = 0
# assert expected_random_theory_threshold == random_theory_threshold


'''
Appendix

[0]
a function which can be applied to the input matrix (network) to determine if it is well-conditioned for the RMT
algorithm would be useful. By well-conditioned, I mean that the matrix must be real-valued, symmetric,
and sufficiently large. 

[1]
"In order to compare eigenvalues of different systems, the mean density
needs to be constant, for example by normalizing to 1 [35]. This is done via spectral
unfolding methods, which map the eigenvalues to a new sequence but maintain some of the
system-specific properties, or universality. Essentially, unfolding maps each eigenvalue λ
to another value λ' in such a way that the resulting unfolded eigenvalue density is constant.
Therefore, without loss of generality, when the NNSD of the eigenvalues is mentioned it
will be implied that a spectral unfolding technique was used to obtain a constant density
of eigenvalues. There are various methods for unfolding the eigenvalues as well, such as
the Gaussian kernel density or a cubic spline interpolation on the cumulative distribution
function"

After calculating the eigenvalues, "The eigenvalue spectrum is then unfolded, i.e. it is calibrated in such a 
way that the average global spacing between the eigenvalues is constant over the whole spectrum. The latter can be 
tracked by setting plot.spacing = T. Two methods are provided for unfolding: one method is based on calculation of
the Gaussian kernel density of the eigenvalue spectrum; another method is based on fitting
a cubic spline function to the cumulative empirical eigenvalue distribution"


[2]
    "The bandwidth here acts as a smoothing parameter, controlling the tradeoff between bias
     and variance in the result. A large bandwidth leads to a very smooth (i.e. high-bias) 
     density distribution. A small bandwidth leads to an unsmooth (i.e. high-variance) density
     distribution." 
     --> https://scikit-learn.org/stable/modules/density.html#kernel-density

[3]
"The point at which the NNSD transitions from the negative exponential
distribution to the Wigner-Dyson distribution marks the threshold of correlation. This means
that everything above that threshold value represents actual correlations in the correlation
matrix and everything below this value is simply static and can be ignored. There should be a
sharp transition the from negative exponential distribution to the Wigner-Dyson distribution"


[4] algorithms parameters which need to be fine-tuned/addressed
"n" bin sizes, domain cut-off (fat-tail distributions more generally), 
bandwidth for kernel (somewhere 0-1), degrees of freedom, 
result stopping value (although 100 seems more or less standard)

'''