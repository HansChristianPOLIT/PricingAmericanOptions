import scipy.sparse
import scipy.sparse.linalg
import pandas as pd
import numpy as np

# inspired by:  https://www.researchgate.net/publication/30758355_Numerical_methods_for_the_valuation_of_financial_derivatives
class AmericanPutFiniteDifference:
    def __init__(self, K: float, r: float, M: int):
        """
        Initialize the AmericanPutOption class.

        Parameters:
        K (float): Strike price of the option.
        r (float): Risk-free interest rate.
        M (int): Number of price steps.
        N (int): Number of time steps.
        """
        self.K = K
        self.r = r
        self.M = M

    def implicit_FD(self, S0, σ, T, N):
        """
        Calculate the American put option price using implicit method.

        Parameters:
        S0 (float): Initial stock price.
        sigma (float): Volatility of the stock.
        T (float): Time to maturity.

        Returns:
        float: Price of the American put option.
        """
        
        # unpack
        M = self.M
        r = self.r
        K = self.K
        
        dt = T / N
        ds = 2 * S0 / M
        A = scipy.sparse.lil_matrix((M+1, M+1))
        f = np.maximum(K - np.arange(0, M+1) * ds, 0)

        for m in range(1, M):
            x = 1 / (1 - r * dt)
            A[m, m-1] = x * (self.r * m * dt - σ**2 * m**2 * dt) / 2
            A[m, m] = x * (1 + σ**2 * m**2 * dt)
            A[m, m+1] = x * (-r * m * dt - σ**2 * m**2 * dt) / 2

        A[0, 0] = 1
        A[M, M] = 1

        for i in range(N, 0, -1):
            f = scipy.sparse.linalg.spsolve(A.tocsr(), f)
            f = np.maximum(f, K - np.arange(0, M+1) * ds)

        P = f[round((M+1) / 2)]
        return P

    def option_pricing(self, combinations, N):
        """
        Calculate prices for different combinations of S, σ, and T.

        Parameters:
        combinations (list of tuples): List of combinations of S, σ, and T.

        Returns:
        list of dicts: Results containing S0, sigma, T, and calculated price.
        """
        results = []
        for combo in combinations:
            S0, σ, T = combo
            price = self.implicit_FD(S0, σ, T, N*T)
            results.append({
                "S0": S0,
                "σ": σ,
                "Maturity": T,
                "Price": price
            })
        return results