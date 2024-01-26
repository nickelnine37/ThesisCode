from ast import Tuple
import numpy as np
import pandas as pd
from scipy.sparse.linalg import LinearOperator, cg


filters = {'diffusion': lambda ln, lt, bn, bt: np.exp(-(bn * ln + bt * lt)),
            '1-hop': lambda ln, lt, bn, bt: (1 + bn * ln + bt * lt) ** -1,
            'relu': lambda ln, lt, bn, bt: np.maximum(1 - (bn * ln + bt * lt), 0),
            'sigmoid': lambda ln, lt, bn, bt: 2 * (1 + np.exp(bn * ln + bt * lt)) ** -1,
            'gaussian': lambda ln, lt, bn, bt: np.exp(-(bn * ln + bt * lt) ** 2),
            'bandlimited': lambda ln, lt, bn, bt: ((bn * ln + bt * lt) <= 1).astype(float)}

def vec(X: np.ndarray) -> np.ndarray:
    return X.T.reshape(-1)


def mat(x: np.ndarray, shape: tuple) -> np.ndarray:
    return x.reshape((shape[1], shape[0])).T


def conjugate_grad(A: LinearOperator, b: np.ndarray, x: np.ndarray=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    
    n = len(b)
    
    if not x:
        x = np.zeros(n)

    r = A.matvec(x) - b
    p = - r
    r_k_norm = np.dot(r, r)

    for i in range(2 * n):

        Ap = A.matvec(p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm

        if r_kplus1_norm < 1e-5:
            break

        p = beta * p - r

    return x, i


class GSR2D:

    def __init__(self, 
                 LT: np.ndarray, 
                 LN: np.ndarray,
                 gamma: float,
                 beta: float | np.ndarray,
                 filter: str='diffusion') -> None:
        """
        Initialise a Graph Signal Reconstruction model. 

        Params:
            LT:         The time-like graph Laplacian. Shape (T, T)
            LN:         The space-like graph Laplacian. Shape (N, N)
            gamma:      The regularisation parameter. Must be >0
            beta:       The graph filter parameter(s). Can be a float or pair of floats. Must be >0
            filter:     Which graph filter to use. Must be one of ['diffusion', '1-hop', 'relu', 'sigmoid', 
                        'gaussian', 'bandlimited'] or a function that takes four inputs: (λn, λt, βn, βt)
        """

        # get signal shape
        self.T = LT.shape[0]
        self.N = LN.shape[0]

        # decompose Laplacians
        self.lamT, self.UT = np.linalg.eigh(LT)
        self.lamN, self.UN = np.linalg.eigh(LN)

        # set graph filter
        if isinstance(filter, str):
            self.filter = filters[filter]
        else:
            self.filter = filter

        # set gamma
        assert gamma > 0
        self.gamma = float(gamma)
        
        # set beta
        if isinstance(beta, float | int):
            assert beta > 0
            self.beta = [beta, beta]
        else:
            assert beta[0] > 0
            assert beta[1] > 0
            self.beta = list(beta)

        
    def predict(self, 
                Y: pd.DataFrame | np.ndarray, 
                method: str='CGM', 
                max_iters: int=1e7) -> pd.DataFrame | np.ndarray:
        """
        Reconstruct a graph signal Y

        Params:
            Y           (N, T) partially observed graph signal. Missing values should be indicated with np.nans
            method      'CGM' or 'SIM'
            max_iters   The maximum number of iterations

        Returns:
            F           (N, T) the posterior mean
            nits        The number of iterations
        """

        if isinstance(Y, pd.DataFrame):
            Y_ = Y.values.copy()
        else:
            Y_ = Y.copy()

        S = ~np.isnan(Y_)
        Y_[~S] = 0
        S = S.astype(float)

        if method == 'CGM':
            F, nits = self.CGM(Y_, S, max_iters)

        elif method == 'SIM':
            F, nits = self.SIM(Y_, S, max_iters)

        if isinstance(Y, pd.DataFrame):
            return pd.DataFrame(F, columns=Y.columns, index=Y.index), nits
        
        return F, nits
        

    def SIM(self, Y: np.ndarray, S: np.ndarray, max_iters: int=1e7) -> tuple[np.ndarray, int]:
    
        """
        Perform the Stationary Iterative Method for computing the posterior mean

        Params:
            Y           (N, T) partially observed graph signal
            S           (N, T) binary sensing matrix
            max_iters   The maximum number of iterations

        Returns:
            F           (N, T) the posterior mean
            nits        The number of iterations
        """

        N, T = Y.shape
        assert N == self.N and T == self.T, f'Y should have shape {(self.N, self.T)} but it has shape {Y.shape}'

        G2 = self.filter(self.lamN[:, None], self.lamT[None, :], self.beta[0], self.beta[1]) ** 2
        J = G2 / (G2 + self.gamma)
        S_ = 1 - S

        dF = self.UN @ (J * (self.UN.T @ Y @ self.UT)) @ self.UT.T
        F = dF

        nits = 0
        while (dF ** 2).sum() ** 0.5 / (N * T) > 1e-8:

            dF = self.UN @ (J * (self.UN.T @ (S_ * dF) @ self.UT)) @ self.UT.T
            F += dF
            nits += 1

            if nits == max_iters:
                print(f'Warning: Maximum iterations ({max_iters}) reached')
                break

        return F, nits


    def CGM(self, 
            Y: np.ndarray, 
            S: np.ndarray, 
            max_iters: int=1e7) -> tuple[np.ndarray, int]:
        """
        Perform the Conjugate Gradient Method for computing the posterior mean

        Params:
            Y           (N, T) partially observed graph signal
            S           (N, T) binary sensing matrix
            max_iters   The maximum number of iterations

        Returns:
            F           (N, T) the posterior mean
            nits        The number of iterations
        """

        N, T = Y.shape
        assert N == self.N and T == self.T, f'Y should have shape {(self.N, self.T)} but it has shape {Y.shape}'

        G = self.filter(self.lamN[:, None], self.lamT[None, :], self.beta[0], self.beta[1])

        def matvec(z):
            Z = mat(z, (N, T))
            out = self.gamma * Z + G * (self.UN.T @ (S * (self.UN @ (G * Z) @ self.UT.T)) @ self.UT)
            return vec(out)
        
        nits = 0

        def iter_count(arr):
            nonlocal nits
            nits += 1
        
        linop = LinearOperator((N * T, N * T), matvec=matvec)
        
        z, exit_code = cg(linop, vec(G * (self.UN.T @ Y @ self.UT)), x0=np.random.normal(size=N * T), callback=iter_count, maxiter=max_iters)

        return self.UN @ (G * mat(z, (N, T))) @ self.UT.T, nits

        # z, nits = conjugate_grad(linop, vec(G * (self.UN.T @ Y @ self.UT)))

        # print(z, nits)

        # raise ValueError

        # return self.UN @ (G * mat(z, (N, T))) @ self.UT.T, nits