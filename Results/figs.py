import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Global settings for LaTeX rendering
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'


############################
### Plot MC Sample Paths ###
############################
def plot_stock_price_simulations(T, n, simulated_prices, num_paths):
    """
    Plot the simulated paths of stock prices over time as well as the mean path.

    Parameters:
    T (float): Time to maturity in years.
    n (int): Total number of discrete time steps in the simulation.
    simulated_prices (numpy.ndarray): Array of simulated stock prices. Each row represents a different simulation path.
    num_paths (int): Number of simulated paths.

    Returns:
    None: This function displays a matplotlib plot.
    """

    # Define the time interval for plotting
    time = np.linspace(0, T, n)

    # Calculate the mean of the simulated prices at each time step
    mean_path = np.mean(simulated_prices, axis=0)

    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot each simulation path
    plt.plot(time, simulated_prices.T, lw=1, alpha=0.25)

    # Plot the mean path for emphasis
    plt.plot(time, mean_path, 'b', lw=2, alpha=0.75, label='Mean Path')

    # Labeling the plot
    plt.xlabel("Time (Years)")
    plt.ylabel("Stock Price")
    plt.title(f'{num_paths:,} Stock Price Simulation Paths')
    plt.legend()

    # Display the plot
    plt.show()

    
#############################
### Plot Truncated Domain ###
#############################
def plot_truncated_domain(domain, chebyshev_polynomials, x_next, output_path):
    """
    Plots the truncated domain for a Chebyshev polynomial-based approach, displaying the knot points.

    Parameters:
    domain (tuple): A tuple (lower_boundary, upper_boundary) defining the lower and upper boundaries of the truncated domain.
    chebyshev_polynomials (int): The number of Chebyshev polynomials, indicating the count of knot points.
    x_next (numpy.ndarray): A 2D array where each column corresponds to the knot points for a given polynomial degree.
    output_path (str): The file path where the plot will be saved. 

    Returns:
    None: The function generates and saves a plot but does not return any value.
    """

    lower_boundary, upper_boundary = domain
    
    plt.figure(figsize=(12, 8))

    for i in range(chebyshev_polynomials):
        plt.scatter([i] * len(x_next[:, i]), x_next[:, i], alpha=0.5, color='black', s=8, marker='s', label=f'Knot {i+1}')

    plt.axhline(y=lower_boundary, color='r', linestyle='-')
    plt.axhline(y=upper_boundary, color='r', linestyle='-')
    plt.grid(False)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlabel(f'Number of $x$ knot points', fontsize=14)
    plt.ylabel(f'xnext coordinates', fontsize=14)
    plt.tick_params(axis='both', direction='in', length=6, width=1, colors='black', grid_alpha=0.5, labelsize=14)
    plt.tight_layout()

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()