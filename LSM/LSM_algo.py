import numpy as np
import math


# OOP Approach 
class AmericanOptionsLSM(object).
    """ Class for pricing American Options using Least-Square Monte Carlo as proposed by Longstaff-Schwartz.

    """
    
    def __init__(self, option_type, S0, strike, T, M, r, div, sigma, itm, simulations):
        try:
            self.option_type = option_type
            assert isinstance(option_type, str)
            self.S0 = float(S0)
            self.strike = float(strike)
            assert T > 0
            self.T = float(T)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = float(r)
            assert div >= 0
            self.div = float(div)
            assert sigma > 0
            self.sigma = float(sigma)
            assert simulations > 0
            self.simulations = int(simulations)
        except ValueError:
            print('Error passing Options parameters')
        
        if option_type != 'call' and option_type != 'put':
            raise ValueError("Error: option type not valid. Enter 'call' or 'put'")
        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)
        
    

def LSM(K, option='call',itm=True):
    """ Valuation of American Options by Least-Square Monte Carlo 
    
    Parameters
    ----------
    itm : boolean 
        if True use only from in the money starts option
        
    Returns
    -------
    V0 : float
        estimated value of American Option
        
    """
    # initialize 
    V = h[-1] # last period is a trivial decision
        
    # LSM algorithm (valuation by backwards induction)
    for t in range(M-1, 1, -1):
            rg = np.polyfit(S[t], V*df, 5)
            C = np.polyval(rg, S[t]) # continuation value
            V = np.where(h[t] < C, h[t], V*df) # exercise decision
    # LSM estimator 
    V0 = df * np.mean(V)
    
    return V0 
    