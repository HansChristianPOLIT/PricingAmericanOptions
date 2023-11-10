import numpy as np

def cartesian_product(*arrays):
    """
    Calculate the cartesian product of a list of input arrays.
    
    Parameters
    ----------
    arrays : 
        a parameter list of 1d arrays

    Returns
    -------
    ndarray of shape (N, D)
        a matrix with columns D equal to number of input arrays and
        rows N equal to the product of number of elements in input
        arrays

    """
    mg = np.meshgrid(*arrays, indexing='ij')
    st = np.stack(mg)
    st = np.transpose(st, list(range(1,len(st.shape))) + [0])
    return np.reshape(st, (-1, len(mg)))


def chebyshev_transform(x, a = -1.0, b = 1.0):
    """
    Transform input x from standard domain [-1, 1] to general
    hyper-rectangular domain.

    Parameters
    ----------
    x : 1d array or 2d array on the standard domain [-1, 1]
    a : float or 1d array
        lower boundary of domain, if a is array then we require a.shape[0]
        equal to x.shape[-1]
    b : float or 1d array
        upper boundary of domain, if b is array then we require b.shape[0]
        equal to x.shape[-1]
    
    Returns
    -------
    1d array or 2d array of shape x.shape    
    """
    return a + 0.5 * (b - a) * (x + 1.0)
    

def chebyshev_inverse_transform(y, a = -1.0, b = 1.0):
    """
    Transform input y from general hyper-rectangular domain to
    standard domain [-1, 1].

    Parameters
    ----------
    y : 1d array or 2d array on general hyper-rectangular domain
    a : float or 1d array
        lower boundary of domain, if a is array then we require a.shape[0]
        equal to y.shape[-1]
    b : float or 1d array
        upper boundary of domain, if b is array then we require b.shape[0]
        equal to y.shape[-1]
    
    Returns
    -------
    1d array or 2d array of shape y.shape    
    """
    return 2 / (b-a) * (y-a) - 1.0


def chebyshev_points(degrees):
    """
    Calculate the Chebyshev points (of second kind) used for interpolation.

    Parameters
    ----------
    degrees : list of int
        each entry represents the maximum polynomial degree per dimension,

    Returns
    -------
    list of 1d arrays


    Chebyshev points of second represent the extrema of Chebyshev polynomials
    of first kind.
    """
    return [ np.cos(np.pi * np.linspace(0, 1, n+1)) for n in degrees]


def chebyshev_multi_points(degrees):
    """
    Calculate multivariate Chebyshev points on a standard domain [-1,1].
    
    Parameters
    ----------
    degrees : list of int
        each entry represents the maximum polynomial degree per dimension,
        len(degrees) corresponds to the number of dimensions
        
    Returns
    -------
    2d array of shape (N,D)
        number of rows N equals product over all (N_d + 1) where N_d is
        the maximum polynomial degree in dimension d (d=1...D) and D is
        the number of dimensions of the tensor.
    """
    points = chebyshev_points(degrees)
    multi_points = cartesian_product(*points)
    return multi_points


def chebyshev_polynomials(x, max_degree):
    """
    Calculate the Chebyshev polynomials T_0(x), ..., T_N(x)
    for up to maximum degree N.

    Parameters
    ----------
    x : float or 1d array
        for interpolation x is assumed to be in [-1, 1] (element-wise)

    Returns
    -------
    1d array if input is float or 2d array if input is array. First
    dimension represents maximum degree N.
    """
    T = [ 1.0 + 0*x ]
    if max_degree==0:
        return np.array(T)
    T += [ x ]
    if max_degree==1:
        return np.array(T)
    for d in range(2, max_degree+1):
        T += [ 2*x*T[-1] - T[-2] ]
    return np.array(T)


def chebyshev_batch_call(C, x):
    """
    Calculate

    z = [...[C * T(x_D)] * ...] * T(x_1)

    for a tensor C and input points x.

    T(x_d) are Chebyshev polynomials to degree N_d.

    Parameters
    ----------
    C : ndarray with suitable shape
        a tensor specifying the maximum polynomial degrees per dimension.

    x : ndarray with shape (N, D)
        a matrix where first dimension N represents batch size and
        second dimension D represents number of dimensions of tensor.
        
    Returns
    -------
    1d array of shape (N,)

    This is the basic operation for calibration and and interpolation.

    Re-shaping is to ensure proper broadcast in matmul.
    """
    degrees = [ d-1 for d in C.shape ]
    assert len(x.shape)==2
    assert x.shape[1]==len(degrees)
    res = np.reshape(C, [1] + list(C.shape))
    for xi, Nd in zip(reversed(x.T), reversed(degrees)):
        T = chebyshev_polynomials(xi, Nd)
        T = np.reshape(T.T, [T.shape[1]] + [1]*(len(res.shape)-3) + [T.shape[0]] + [1])
        if len(res.shape)==2: # last iteration
            res = np.reshape(res, [res.shape[0]]+[1]+[res.shape[1]])
        res = np.matmul(res, T)
        res = np.reshape(res, res.shape[:-1])
    return np.reshape(res, (-1))


def chebyshev_coefficients(degrees, multi_points, values):
    """
    Calculate coefficients of Chebyshev basis functions.

    Parameters
    ----------
    degrees : list of int
        each entry represents the maximum polynomial degree per dimension,
        len(degrees) corresponds to the number of dimensions
    multi_points : 2d array
        multivariate Chebyshev points on a standard domain [-1,1]
    values : 1d array
        the target function values for each D-dimensional multivariate
        Chebyshev point on the transformed domain of the tagret function.
        
    Returns
    -------
    ndarray of shape (degrees + 1)
    """
    assert len(multi_points.shape) == 2
    assert len(values.shape) == 1
    assert multi_points.shape[0] == np.prod([n+1 for n in degrees])
    assert multi_points.shape[0] == values.shape[0]
    assert multi_points.shape[1] == len(degrees)
    #
    idx_list = [ np.arange(n+1) for n in degrees]
    multi_idx = cartesian_product(*idx_list)
    inner_node_factor = np.sum(
        (0<multi_idx) & (multi_idx<np.array(degrees)),
        axis = 1
    )
    values_adj = values * 0.5**(len(degrees) - inner_node_factor)
    values_adj = np.reshape(values_adj, [n+1 for n in degrees])  # as tensor
    multi_coeff = chebyshev_batch_call(values_adj, multi_points)
    multi_coeff *= 2**inner_node_factor / np.prod(degrees)
    coeff = np.reshape(multi_coeff, [n+1 for n in degrees]) # as tensor 
    return coeff


def chebyshev_interpolation(y, coeff, a = -1.0, b = 1.0):
    """
    Calcuate multivariate Chebyshev interpolation.

    Parameters
    ----------
    y : 2d array
        Inputs of shape (N, D) where N is batch size and D is the number
        of dimensios. Input is assumed from general hyper-rectangular
        domain
    coeff : ndarray
        calibrated Chebyshev tensor coefficients
    a : float or 1d array
        lower boundary of domain, if a is array then we require a.shape[0]
        equal to y.shape[-1]
    b : float or 1d array
        upper boundary of domain, if b is array then we require b.shape[0]
        equal to y.shape[-1]
    
    Returns
    -------
    1darray of shape (N,)
    """
    x = chebyshev_inverse_transform(y, a, b)
    return chebyshev_batch_call(coeff, x)
