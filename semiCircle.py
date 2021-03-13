import random
import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)  # scientific notation off
pi = math.pi


def _gen_random_data(row_len=1500, col_len=1500, seed=5, df_type='int'):
    """
        Generate completely random data.
    """
    if seed is not None:
        np.random.seed(seed)

    idx_labels = ['row_{}'.format(i) for i in range(row_len)]
    col_labels = ['col_{}'.format(i) for i in range(col_len)]

    if df_type == 'int':
        random_df = pd.DataFrame(np.random.randint(-100, 100, size=(row_len, col_len)),
                                 index=idx_labels,
                                 columns=col_labels)
    elif df_type == 'float':
        random_df = pd.DataFrame(np.random.rand(row_len, col_len),
                                 index=idx_labels,
                                 columns=col_labels)
    else:
        raise ValueError('unexpected df type')

    return random_df


def _gen_related_data(n, df_type='int'):   # needs work
    """
       Generate data with linear relationships present.
    """
    arr = []
    for i in range(n):
        arr.append([])
        for j in range(n):
            signal = int(i/5)   # small divisor --> less random,  large divisor --> more random
            arr[i].append(random.randrange(-100, 100+signal))

    idx_labels = ['row_{}'.format(i) for i in range(n)]
    col_labels = ['col_{}'.format(i) for i in range(n)]

    if df_type == 'int':
        random_df = pd.DataFrame(arr,
                                 index=idx_labels,
                                 columns=col_labels)
    elif df_type == 'float':
        random_df = pd.DataFrame(arr,
                                 index=idx_labels,
                                 columns=col_labels)
    else:
        raise ValueError('unexpected df type')

    return random_df      # needs


def _gen_correlation_matrix(matrix):
    """
        Generate correlation statistics for given data
    """
    print("----------------------------------------------------------------------------------")
    correlation_matrix = matrix.corr(method='pearson')
    x = np.asarray(correlation_matrix)
    print("Correlation Data:")
    print("mean:", np.round(x.mean(), 3), "var:", np.round(x.std(), 3))
    pos = x[x > 0]
    neg = x[x < 0]

    print("pos:", pos.size, ", neg: ", neg.size, ", tot:", x.size)
    print(np.round(correlation_matrix, 2))
    return correlation_matrix


def normalize_entries(corr_df):
    """
        Normalize given matrix to 0 mean 1 variance
    """
    '''Numpy way: faster'''
    # x = np.asarray(corr_df)
    # if x.std() != 1.0 and x.mean() != 0.0:
    #     return np.round((x - x.mean()) / x.std(), 4)
    # elif x.std() != 1.0:
    #     return np.round(x / x.std(), 4)
    # elif x.mean() != 0.0:
    #     return np.round(x - x.mean(), 4)
    # else:
    #     return corr_df

    '''Panda way: prettier'''
    # copy the data frame
    df_std = corr_df.copy()

    # population standard deviation with Pandas
    df_std.std(ddof=1)

    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    corr_df = df_std
    return corr_df


def normalize_eigenvalues(eigen):
    """
         Normalize eigenvalues
    """
    return [eig / math.sqrt(len(eigen)/2) for eig in eigen]
    # return eigen/math.sqrt(len(eigen)/2)  # numpy array


def plot(eigen_value):  # random entries should follow semi-circle law as n --> infinity
    """
        Histogram the eigenvalues, plot distribution.
    """
    # Semi-Circle Distribution
    # x = np.arange(-2., 2., 0.0001)
    # circle = []
    # for i in range(0, len(x)):
    #     y = 1/(2*pi)*math.sqrt(4 - x[i]*x[i])  # disc equation
    #     circle.append(y)

    # Plot results
    # plt.plot(x, circle, 'r')
    y1, x1, _ = plt.hist(eigen_value, bins=40, density=True)
    # print(x1.max())
    # print(y1)
    plt.title("Semi-Circle Law")
    plt.ylabel("Density")
    plt.xlabel("Eigenvalues")
    # plt.axis([-2.25, 2.25, 0, 1.])  # manual window
    plt.show()


def _semi_circle(corr_df):
    """
    Takes on input a matrix, normalizes, passes to plot function.
    """
    x = np.asarray(corr_df)
    print("Initial Data:")
    print("mean:", np.round(x.mean(), 3), "var:", np.round(x.std(), 3))
    pos = x[x > 0]
    neg = x[x < 0]

    print("pos:", pos.size, ", neg: ", neg.size, ", tot:", x.size)

    print(corr_df)
    # eigen_init = np.linalg.eigvalsh(x)
    # print("... with initial eigenvalues", "\n", eigen_init, "\n")

    print("----------------------------------------------------------------------------------")

    # mask = np.ones(x.shape, dtype=bool)
    # np.fill_diagonal(mask, 0)      # ignore diagonals while finding max
    # max_value = abs(x[mask]).max()
    # t = max_value
    # print("max eigenvalue:", t)

    norm_df = normalize_entries(corr_df)
    x = np.asarray(norm_df)
    print("Normalized Data:")
    print("mean:", np.round(x.mean(), 3), "variance:", np.round(x.std(), 3))

    pos = x[x > 0]
    neg = x[x < 0]
    print("pos:", pos.size, ", neg: ", neg.size, ", tot:", x.size)
    print(np.round(norm_df, 2))

    corr_df = _gen_correlation_matrix(norm_df)

    # Gaussian Orthogonal Ensemble: M = ½(A + AT):
    trans = np.transpose(norm_df)       # = AT
    h = np.add(norm_df, trans)/2        # make symmetric
    eigen_goe = np.linalg.eigvalsh(h)
    # print("GOE eigenvalues:", eigen_goe, "\n")

    norm_goe = normalize_eigenvalues(eigen_goe)
    plot(norm_goe)

    threshold = 0
    return threshold


def _compute_threshold_value(matrix):
    # todo cannot begin until we know the expected amount of signals.
    t = 1.
    return t


"""
    Run and Test
"""

# seed = 4

# random_theory_threshold = _semi_circle(_gen_random_data())
random_theory_threshold = _semi_circle(_gen_related_data(1500, df_type='int'))


# expected_random_theory_threshold = 0
# assert expected_random_theory_threshold == random_theory_threshold

