import numpy as np
from numba import njit
import scipy as sp
def sparse_matrix_vector_multiplication(rows, cols, values, vector):

    result = np.zeros(len(vector))

    for i in range(0, len(values)):
        result[rows[i]] += values[i] * vector[cols[i]]

    print("Result:", result)


if __name__ == "__main__":
    num_values = 10000
    rng = np.random.default_rng()
    rvs = sp.stats.poisson(25, loc=10).rvs
    S = sp.sparse.random_array((num_values, num_values), density=0.25, random_state=rng, data_sampler=rvs)

    rows = np.array(S.coords[0])
    cols = np.array(S.coords[1])
    values = np.array(S.data)
    vector = np.random.randint(low=1, high=10, size=num_values)

    numba_sparse_matrix_vector_multiplication = njit(sparse_matrix_vector_multiplication)
    numba_sparse_matrix_vector_multiplication(rows, cols, values, vector)
    sparse_matrix_vector_multiplication(rows, cols, values, vector)
