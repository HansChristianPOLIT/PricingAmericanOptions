import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

class DynamicChebyshev:
    def __init__(self, r, S0: float, K: float, T: float, σ: float, λ: float,
                 dim: int, n: int, n_chebyshev_pol: int, seed: int, use_AV: bool = False):
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
        n_chebyshev_pol (int): degree of chebyshev polynomials
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
        self.λ = λ
        self.n = n
        self.n_chebyshev_pol = n_chebyshev_pol 
        self.n_chebyshev_point = self.n_chebyshev_pol + 1 
        self.V = np.zeros((self.n_chebyshev_point, self.n))
        self.C = np.zeros((self.n_chebyshev_point, self.n-1))
        self.dim = dim
        self.Δ = self.T / self.n
        self.x_next = np.zeros((self.dim, self.n_chebyshev_point))
        self.Γ = np.zeros((self.n_chebyshev_point, self.n_chebyshev_point)) # generalized moment
        self.chebypol_eval = np.zeros((self.dim, self.n_chebyshev_point, self.n_chebyshev_point))
        self.use_AV = use_AV
        
        if use_AV:
            assert dim % 2 == 0, 'For AV, the number of paths ("dim") must be even'
            half_dim = self.dim // 2
            Z_half = np.random.normal(0, 1, (half_dim, self.n_chebyshev_point)) #Z_half matrix with dimension (half_dim, self.n-1), representing random increments of the asset´s price over time for half the paths'
            self.Z = np.concatenate((Z_half, -Z_half), axis=0)  # Antithetic variates. Creating full matrix self:Z by concatenating Z_half with its negation -Z.half.
        else:
            self.Z = np.random.normal(0, 1, (self.dim, self.n_chebyshev_point))
            
        # Generate Poisson and (log-)normal random jumps for all paths and time steps at once
        self.N = np.random.poisson(self.λ*self.Δ, (self.dim, self.n_chebyshev_point))  # Poisson process for the number of jumps
        self.Z_2 = np.random.normal(0, 1, (self.dim, self.n_chebyshev_point))  # Normal random variables for the jump sizes

    def calculate_truncated_domain_GBM(self):
        """ 
        Defines truncated general domain, χ, for GBM.

        Returns:
        list: A list containing the lower and upper bounds of the truncated domain.
        """
        
        # unpack 
        S0 = self.S0
        r = self.r
        T = self.T
        σ = self.σ

        μ = np.log(S0) + (r - 0.5*σ**2)*T
        trunc = 3 * np.sqrt(T) * σ # define truncation
        χ = [μ - trunc, μ + trunc] # truncated domain

        return χ
    
    def calculate_truncated_domain_JumpMerton(self, α: float, β: float):
        """ 
        Defines truncated general domain, χ, for Jump Merton.
        
        Parameters:
        α (float): Mean of log-normal jump size
        β (float): Volatility of log-normal jump size

        Returns:
        list: A list containing the lower and upper bounds of the truncated domain.
        """
        
        # unpack 
        S0 = self.S0
        r = self.r
        T = self.T
        σ = self.σ
        λ = self.λ

        μ = np.log(S0) + (r - 0.5*σ**2 - λ*(np.exp(α + 0.5*β**2) - 1))*T
        trunc = 12 * np.sqrt(T) * σ # define truncation
        χ = [μ - trunc, μ + trunc] # truncated domain

        return χ
    
    def chebyshev_points_1d(self,degree: int):
        """
        Calculate the chebyshev points used for one-dimensional interpolation.

        Parameters:
        degrees (int): degree of chebyshev polynomial.

        Returns:
        1d array
        """
        
        # unpack 
        n_chebyshev_pol = self.n_chebyshev_pol
        
        # x_i = (j * π / N) for i in range 0,...,N
        return np.cos(np.pi * np.linspace(0, 1, n_chebyshev_pol+1))

    def chebyshev_transform_1d(self,z, a: float, b: float):
        """
        Transform input z from standardized domain [-1,1] to general domain χ.

        Parameters:
        z (1d array): chebyshev points on standardized domain [-1,1]
        a (float): lower boundary on domain
        b (float): upper boundary on domain

        Returns: 
        1d array of shape z.shape on χ.
        """

        return b + 0.5*(a-b)*(1.0-z)

    def chebyshev_inverse_transform_1d(self, x, a: float, b: float):
        """
        Transform input x from general domain χ to standardized domain [-1,1].

        Parameters:
        x (1d array): chebyshev points on general domain χ
        a (float): lower boundary of general domain
        b (float): upper boundary of general domain

        Returns:
        1d array of shape x.shape
        """

        return 2*(x-a)/(b-a) - 1
    
    def calculate_nodal_points(self, χ):
        """ 
        Nodal points.
        
        Parameters:
        χ (list): The truncated general domain.

        Returns:
        ndarray: A 1D array of nodal points in the domain.
        """
        
        # unpack 
        n_chebyshev_pol = self.n_chebyshev_pol
        
        # Nodal points
        z = self.chebyshev_points_1d(n_chebyshev_pol)  # standardized domain [-1,1]
        xknots = self.chebyshev_transform_1d(z, χ[0], χ[1])  # transform to hyperrectangular domain

        return xknots

    def generate_GBM_path(self, xknots):
        """ 
        Compute Monte Carlo paths using Geometric Brownian Motion under risk-neutral measure.
        
        Parameters:
        xknots (numpy.ndarray): Initial values of the asset at each point in the Chebyshev grid.

        Returns:
        numpy.ndarray: Simulated asset paths.
        """
        
        # unpack
        Δ = self.Δ
        σ = self.σ
        Z = self.Z
        r = self.r 
        n_chebyshev_point = self.n_chebyshev_point
        x_next = self.x_next
        
        # simulate stock process
        for i in range(n_chebyshev_point):
            x_next[:,i] = xknots[i] + (r-0.5*σ**2)*Δ + σ*np.sqrt(Δ)*Z[:,i]
        
        return x_next
        
    def generate_Jump_path(self, xknots, α: float, β: float):
        """
        Generates paths for the Merton Jump Diffusion model assuming log-normal distribution of shocks.

        Parameters:
        xknots (numpy.ndarray): Initial values of the asset at each point in the Chebyshev grid.
        α (float): Mean of log-normal jump size.
        β (float): Volatility of log-normal jump size.

        Returns:
        numpy.ndarray: Simulated paths of the asset price.
        """
        
        # unpack 
        Δ = self.Δ
        σ = self.σ
        λ = self.λ
        Z = self.Z
        Z_2 = self.Z_2
        N = self.N
        r = self.r 
        n_chebyshev_point = self.n_chebyshev_point
        x_next = self.x_next
        dim = self.dim
        
        # drift corrected term
        c = r - 0.5*σ**2 - λ*(np.exp(α + 0.5*β**2) - 1)
        
        # Calculate the jump sizes for all paths and time steps
        M = α * N + β*np.sqrt(N)*Z_2
        
        # if no jump set M = 0
        M = np.where(N > 0, M, 0)
        
        # Calculate the combined diffusion and jump process for all knots
        for i in range(n_chebyshev_point):
            x_next[:,i] = xknots[i] + c*Δ + σ*np.sqrt(Δ)*Z[:,i] + M[:,i]
    
        return x_next
    
    def generate_CEV_path(self, xknots, γ: float):
        """ 
        Computes Monte Carlo paths using the Constant Elasticity of Variance (CEV) model.

        Parameters:
        xknots (numpy.ndarray): Initial values of the asset at each point in the Chebyshev grid.
        γ (float): Parameter governing elasticity with respect to price.

        Returns:
        numpy.ndarray: Simulated asset paths according to the CEV model.
        """
        
        # unpack
        Δ = self.Δ
        σ = self.σ
        Z = self.Z
        r = self.r 
        n_chebyshev_point = self.n_chebyshev_point
        x_next = self.x_next

        # simulate stock process (note, CEV is the log-dynamics due to how DC is setup!)
        for i in range(n_chebyshev_point):
            x_next[:,i] = xknots[i] + (r-0.5*σ**2*xknots[i]**(γ-2))*Δ + σ*xknots[i]**(γ*0.5 -1)*np.sqrt(Δ)*Z[:,i] 
                                    
        return x_next
        
    def compute_generalized_moments(self, χ, xknots):
        """ 
        Computes generalized moments using Monte Carlo simulation.

        Parameters:
        χ (list): The truncated general domain.
        xknots (numpy.ndarray): Nodal points in the domain.

        Returns:
        numpy.ndarray: A 2D array of generalized moments.
        """
        
        # unpack
        Δ = self.Δ
        σ = self.σ
        Z = self.Z
        r = self.r 
        n_chebyshev_point = self.n_chebyshev_point
        x_next = self.x_next
        Γ = self.Γ 
        chebypol_eval = self.chebypol_eval
        
        # step 1: check if simulate stock process is within domain
        check = (x_next > χ[0]) & (x_next < χ[1]) # indicator function
        valid_points = np.sum(check, axis=0)
        
        # step 2: compute generalized moments
        for j in range(self.n_chebyshev_point):
            y = np.zeros(self.n_chebyshev_point)
            y[j] = 1
            cheb_fun = interp1d(xknots, y, kind='linear', fill_value="extrapolate")
            chebypol_eval = cheb_fun(x_next)*check
            Γ[:,j] = np.sum(chebypol_eval, axis=0) / valid_points
        
        ### OLD; non-vectorized ###
        # step 2: interpolate over domain
        #for j in range(n_chebyshev_point):
        #    y = np.zeros(n_chebyshev_point)
        #    y[j] = 1
        #    cheb_fun = interp1d(xknots, y, kind='linear', fill_value="extrapolate")
        #    chebypol_eval[:,:,j] = cheb_fun(x_next)*check
        
        # step 3: compute generalized moments
        #for i in range(n_chebyshev_point):
        #    for j in range(n_chebyshev_point):
        #        Γ[i,j] = np.sum(chebypol_eval[:,i,j])/valid_points[i]
        
        return Γ
    
    def price_option_with_dynamic_chebyshev(self, xknots, Γ):
        """ 
        Prices an option using backward induction based on Chebyshev polynomial approximations.

        Parameters:
        xknots (numpy.ndarray): Nodal points in the domain.
        Γ (numpy.ndarray): Generalized moments.

        Returns:
        float: The option price at time zero.
        """
        
        # unpack
        n_chebyshev_point = self.n_chebyshev_point
        n = self.n
        K = self.K
        Δ = self.Δ
        r = self.r
        S0 = self.S0
        V = self.V
        C = self.C
        dim = self.dim
        
        # step 1: terminal period, t=T
        payoff = np.maximum(K - np.exp(xknots), 0) # remember to exponentiate since we have form S = e^{x}
        V[:,-1] = payoff

        # step 2: Backward induction
        for i in range(n-2, -1, -1):
            C[:,i] = np.exp(-r*Δ)*Γ@V[:,i+1] # discounted next-period continuation value
            V[:,i] = np.where(payoff>C[:,i], payoff, C[:,i])
            #V[:,i] = np.max(np.column_stack((payoff, C[:,i], np.zeros_like(payoff))), axis=1)
        
        # step 3: time zero, t=0
        C0 = np.exp(-r*Δ)*Γ@V[:,0]
        self.V0 = interp1d(xknots, C0, kind='linear', fill_value="extrapolate")(np.log(S0))
        
        # Calculate the standard error
        V_variance = np.var(V)
        self.standard_error = np.sqrt(V_variance / dim)
        
        return self.V0, self.standard_error