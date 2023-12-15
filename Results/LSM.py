import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import factorial 

class MonteCarloOptionPricing:
    def __init__(self, r, S0: float, K: float, T: float, σ: float,
                 λ: float, dim: int, n: int, seed: int, use_AV: bool = False):
        """ 
        Class for pricing American OptionsLSM. 
        
        Parameters: 
        S0 (float): Initial asset value
        K (float): strike price
        T (float): time to maturity, in years, a float number
        r (float): risk-free interest rate
        σ (float): volatility coefficient for diffusion
        λ (float): Intensity rate of the Poisson process
        dim (int): number of paths to simulate
        n (int): between time 0 and time T, the number of time steps
        use_AV (bool): Flag to use Antithetic Variates method (default: False)
        """
        
        assert σ >= 0, 'volatility cannot be less than zero'
        assert S0 >= 0, 'initial stock price cannot be less than zero'
        assert T >= 0, 'time to maturity cannot be less than zero'
        assert n >= 0, 'no. of slices per year cannot be less than zero'
        assert dim >= 0, 'no. of simulation paths cannot be less than zero'
        
        # Set the random seed for reproducibility
        np.random.seed(seed)
        
        self.r = r
        self.S0 = S0
        self.K = K
        self.T = T
        self.σ = σ
        self.n = n
        self.λ = λ
        self.dim = dim
        self.Δ = self.T / self.n
        self.df = np.exp(-self.r*self.Δ)
        self.use_AV = use_AV
        
        ### Generate the Sources of Randomness ###
        if use_AV:
            assert dim % 2 == 0, 'For AV, the number of paths ("dim") must be even'
            half_dim = self.dim // 2
            Z_half = np.random.normal(0, 1, (half_dim, self.n - 1)) #Z_half matrix with dimension (half_dim, self.n-1), representing random increments of the asset´s price over time for half the paths'
            self.Z = np.concatenate((Z_half, -Z_half), axis=0)  # Antithetic variates. Creating full matrix self:Z by concatenating Z_half with its negation -Z.half.
        else:
            self.Z = np.random.normal(0, 1, (self.dim, self.n - 1))  # Original method
        self.S = np.full((self.dim, self.n), np.nan)  # Allocate space for stock price process, with an extra step for initial value
        
        # Generate Poisson and (log-)normal random jumps for all paths and time steps at once
        self.N = np.random.poisson(self.λ*self.Δ, (self.dim, self.n-1))  # Poisson process for the number of jumps#
        self.Z_2 = np.random.normal(0, 1, (self.dim, self.n-1))  # Normal random variables for the jump sizes

    def GeometricBrownianMotion(self):
        """ 
        Generate GBM paths according to Algorithm 3.
        
        Returns:
        np.ndarray: Simulated paths of the asset price.
        """
        
        # unpack 
        Δ = self.Δ
        Z = self.Z
        S = self.S
        S0 = self.S0
        r = self.r
        σ = self.σ
        n = self.n
        
        S[:,0] = np.log(S0)  # Set initial values
        for j in range(1,n):
            S[:,j] = S[:,j-1] + (r-0.5*σ**2)*Δ + σ*np.sqrt(Δ)*Z[:,j-1]
            
        self.S = np.exp(S)  # Exponentiate to get the GBM paths
        return self.S
    
    ##########################
    ### Vectorized Version ###
    ##########################
    def GeometricBrownianMotion_vec(self):
        """ 
        Generate GBM paths according to Algorithm 3.

        Returns:
        np.ndarray: Simulated paths of the asset price.
        """
        
        # unpack 
        Δ = self.Δ
        Z = self.Z
        S = self.S
        S0 = self.S0
        r = self.r
        σ = self.σ
        n = self.n

        # Generate all increments at once
        BM = (r - 0.5*σ**2)*Δ + σ*np.sqrt(Δ)*Z
        
        # Use cumsum to calculate the cumulative sum of increments and then exponentiate
        S[:,:] = np.log(S0)
        S[:,1:] += np.cumsum(BM, axis=1)

        # Multiply every path by the initial stock price
        self.S = np.exp(S)
        return self.S
    
    def MertonJumpDiffusion(self, α: float, β: float):
        """
        Generate Merton Jump Diffusion paths according to Algorithm 4 assuming log-normal distribution of shocks.
        Parameters:
        α (float): Mean of log-normal jump size
        β (float): Volatility of log-normal jump size
        λ (float): Intensity rate of the Poisson process
        
        Returns:
        np.ndarray: Simulated paths of the asset price
        """
        
        # unpack parameters
        Δ = self.Δ
        Z = self.Z
        Z_2 = self.Z_2
        N = self.N
        λ = self.λ
        S = self.S
        S0 = self.S0
        r = self.r
        σ = self.σ
        n = self.n
        dim = self.dim

        S[:,0] = np.log(S0) 
        c = r - 0.5*σ**2 - λ*(np.exp(α + 0.5*β**2) - 1)
        
        for j in range(1,n):
            # Compute jump sizes for each path
            M = α*N[:,j-1] + β*np.sqrt(N[:,j-1])*Z_2[:,j-1]
            # if no jump set jump process to zero 
            M = np.where(N[:,j-1] > 0, M, 0)
            # Calculate the combined diffusion and jump process
            S[:,j] = S[:,j-1] + c*Δ + σ*np.sqrt(Δ)*Z[:,j-1] + M
            
        self.S = np.exp(S) 
    
        return self.S
    
    ##########################
    ### Vectorized Version ###
    ##########################
    def MertonJumpDiffusion_vec(self, α: float, β: float):
        """
        Generate Merton Jump Diffusion paths according to Algorithm 4 assuming log-normal distribution of shocks.
        Parameters:
        α (float): Mean of log-normal jump size
        β (float): Volatility of log-normal jump size
        
        Returns:
        np.ndarray: Simulated paths of the asset price
        """
        
        # unpack 
        Δ = self.Δ
        Z = self.Z
        Z_2 = self.Z_2
        N = self.N
        λ = self.λ
        S = self.S
        S0 = self.S0
        r = self.r
        σ = self.σ
        n = self.n
        dim = self.dim
        
        # drift corrected term
        c = r - 0.5*σ**2 - λ*(np.exp(α + 0.5*β**2) - 1)
        
        # Calculate the jump sizes for all paths and time steps
        M = α * N + β*np.sqrt(N)*Z_2
        
        # if no jump set M = 0
        M = np.where(N > 0, M, 0)
        
        # Calculate the combined diffusion and jump process for all time steps
        S[:,:] = np.log(S0)
        S[:,1:] = np.log(S0) + np.cumsum(c*Δ + σ*np.sqrt(Δ)*Z + M, axis=1)

        self.S = np.exp(S)

        return self.S

    def CEV(self, γ: float):
        """
        Generate CEV paths according to Algorithm 5. 
        
        Parameters:
        γ (float): parameter governing elasticity with respect to price
        
        Returns:
        np.ndarray: Simulated paths of the asset price
        """ 
        assert γ>= 0, 'Cannot let elasticity be negative due to leverage effect'

        # unpack 
        Δ = self.Δ
        Z = self.Z
        S = self.S
        S0 = self.S0
        r = self.r
        σ = self.σ
        n = self.n

        S[:,0] = S0  # Set initial values
        # Simulation using the Euler-Maruyama method for the CEV model
        for j in range(1,n):
            S[:,j] = S[:,j-1] + r*S[:,j-1]*Δ + σ*S[:,j-1]**(γ/2)*np.sqrt(Δ)*Z[:,j-1]
        self.S = S

        return self.S
    
    def BS_option_value(self, otype: str = 'put'):
        """ 
        Closed form non-dividend valuation of a European option in Black-Scholes.
        
        Parameters:
        otype (str): Option type either call or put (default: put)
        
        Returns:
        float: Option price of a European put option
        """
        
        # unpack 
        S0 = self.S0
        K = self.K
        r = self.r
        σ = self.σ
        T = self.T

        d1 = (np.log(S0/K) + (r + 0.5*σ**2)*T) / (σ*np.sqrt(T))
        d2 = d1 - σ*np.sqrt(T)
        
        if otype == 'call':
            value = (S0 * stats.norm.cdf(d1, 0., 1.) -
                 K * np.exp(-r * T)*stats.norm.cdf(d2, 0., 1.))
        elif otype == 'put':
            value = K * np.exp(-r*T)*stats.norm.cdf(-d2) - S0*stats.norm.cdf(-d1)
        else: 
            raise ValueError('Invalid option type.')
    
        return value
    
    def merton_jump_option_value(self, α, β, max_iter=100 , tol=1e-15):
        """ 
        Semi-closed form valuation of Merton's Log-Normal Jump-Diffusion Model for a European put option.
        
        Parameters:
        α (float): Mean of log-normal jump size
        β (float): Volatility of log-normal jump size
        max_iter (int): Maximum number of iterations for the series expansion. Default is 100.
        tol (float): Stopping condition for the series expansion. Default is 1e-15.
        
        Returns:
        float: Option price of a European put option under Jump Difussion
        """
        
        # unpack
        S0 = self.S0
        K = self.K
        r = self.r
        σ = self.σ
        T = self.T
        λ = self.λ
        
        value = 0
        max_iter = int(max_iter)  # Ensure max_iter is an integer
        for k in range(max_iter):
            r_k = r - λ*(np.exp(α + 0.5*β**2)-1) + (k*(α + 0.5*β**2)) / T
            σ_k = np.sqrt(σ**2 + (k* β**2) / T)
            
            # BS put option
            d1 = (np.log(S0/K) + (r_k + 0.5*σ_k**2)*T) / (σ_k*np.sqrt(T))
            d2 = d1 - σ_k*np.sqrt(T)
            BS_value = K * np.exp(-r_k*T)*stats.norm.cdf(-d2) - S0*stats.norm.cdf(-d1)
            
            # Loop to find semi-closed solution
            sum_k = (np.exp(-(np.exp(α + 0.5*β**2))*λ*T) \
                    * ((np.exp(α + 0.5*β**2))*λ*T)**k / (factorial(k))) * BS_value
            value += sum_k
            if sum_k < tol: 
                return value
        return value # return the value of the option when the maximum value of k is reached
    
    def american_option_LSM(self, poly_degree: int, otype: str = 'put'):
        """
        American option pricing using the LSM as outlined in Algorithm 1.
        
        Parameters:
        poly_degree (int): x^n, number of basis functions
        otype (str): call or put (default)
        
        Returns:
        float: V0, LSM Estimator
        """
        
        assert otype == 'call' or otype == 'put', 'Invalid option type.'
        assert len(self.S) != 0, 'Please simulate a stock price process.'
        
        # unpack
        S = self.S
        K = self.K
        n = self.n
        dim = self.dim
        df = self.df
        T = self.T
        Δ = self.Δ
        
        # Initialize exercise_times array to store exercise times for each path
        self.exercise_times = np.full(dim, T)  # Initialize with T (no exercise)
        
        # Initialize an array to store payoffs
        self.payoffs = np.zeros(dim)
        
        # inner values
        if otype == 'call':
            self.intrinsic_val = np.maximum(S - K, 0)
        elif otype == 'put':
            self.intrinsic_val = np.maximum(K - S, 0)
            
        # last day cashflow == last day intrinsic value
        V = np.copy(self.intrinsic_val[:,-1])
        
        # Backward Induction
        for i in range(n - 2, 0, -1): # start at second to last and end at second to first
            # a. find itm path 
            # (potentially) better estimate the continuation value
            itm_path = np.where(self.intrinsic_val[:,i] > 0)  # evaluate: S[:,i] vs. K
            V = V * df # discount next period value
            V_itm = V[itm_path[0]] # define subset (note, we need to set [0] due to np.where being tuple)
            S_itm = S[itm_path[0],i]
            
            # b. run regression and calculate conditional expectation (LSM)
            # initialize continuation value
            self.C = np.zeros(shape=dim)
            C = self.C
            # if only 5 itm paths (probably, otm options), then continuation value is zero
            if len(itm_path[0]) > 5:
                rg = np.polyfit(S_itm, V_itm, poly_degree)  # polynomial regression
                C[itm_path[0]] = np.polyval(rg, S_itm)  # evaluate conditional expectation
            
            # c. Calculation of value function at i 
            # if hold: V = 0, if exercise: V = intrinsic value
            exercise_condition = self.intrinsic_val[:, i] > C
            V = np.where(exercise_condition, self.intrinsic_val[:, i], V)
            
            for idx in np.where(exercise_condition)[0]:
                self.payoffs[idx] = self.intrinsic_val[idx, i]
                self.exercise_times[idx] = i * Δ
            
            # Update exercise times for paths that exercised
            self.exercise_times[exercise_condition] = i * Δ
            
        self.V0 = df * np.average(V)
        
        # Calculate the standard error
        V_variance = np.var(V)
        self.standard_error = np.sqrt(V_variance / dim)
        
        return self.V0, self.exercise_times, self.payoffs, self.standard_error