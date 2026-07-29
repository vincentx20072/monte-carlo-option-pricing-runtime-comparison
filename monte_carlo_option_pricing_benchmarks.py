"""
Monte Carlo European call option pricer in :
1) Pure Python
2) NumPy Vectorize 
3) Numba JIT
"""

#Dependencies
import math
import time
import numpy as np
from numba import njit,prange
import matplotlib.pyplot as plt

# Pure Python simulator
def mc_call_pure_python(s0,k,r,sigma,t,normals):
    """
    Parameters for this function:
    s0 (float) : Current stock price
    k (float) : Strike price
    r (float) : Risk-free interest rate
    sigma (float) : Volatility
    t (float) : Time to maturity (years)
    normals (list of floats) : Simulated N(0,1) random variables
    """

    # Pre-compute the constant terms in the Geometric Brownian Motion solution.
    drift = (r - 0.5 * sigma**2) * t
    vol_sqrt_t = sigma * math.sqrt(t)

    payoff_sum = 0.0

    for z in normals:
        # Simulate the stock price at maturity using the closed-form solution of Geometric Brownian Motion.
        st = s0 * math.exp(drift + vol_sqrt_t * z)

        # European call option payoff formula
        payoff = max(st - k, 0.0)
        payoff_sum += payoff

    average_payoff = payoff_sum / len(normals)

    # Discount the expected payoff to obtain the option price.
    option_price = math.exp(-r * t) * average_payoff

    return option_price

# NumPy Vectorize simulator
def mc_call_numpy(s0, k, r, sigma, t, normals):
    """ 
    Note, normals is now a NumPy array of floats
    """

    drift = (r - 0.5 * sigma * sigma) * t
    vol_sqrt_t = sigma * math.sqrt(t)

    st = s0 * np.exp(drift + vol_sqrt_t * normals)
    average_payoffs = np.maximum(st - k, 0.0)

    option_price = math.exp(-r * t) * float(np.mean(average_payoffs))

    return option_price

# Numba JIT simulator
@njit(cache=True, fastmath=True, parallel=True)
def mc_call_numba(s0, k, r, sigma, t, normals):
    """
    Numba JIT version with parallelised loop over Monte Carlo paths.
    """
    drift = (r - 0.5 * sigma * sigma) * t
    vol_sqrt_t = sigma * math.sqrt(t)

    payoff_sum = 0.0

    for i in prange(normals.shape[0]):
        st = s0 * math.exp(drift + vol_sqrt_t * normals[i])
        payoff = st - k
        if payoff > 0.0:
            payoff_sum += payoff

    option_price = math.exp(-r * t) * payoff_sum / normals.shape[0]

    return option_price

# Function execution timer
def time_call(fn, *args):
    start = time.perf_counter()
    value = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms

def compare_runtime(s0,k,r,sigma,t,path_sizes,seed=123,n_repeats=5):
    """
    Measure how runtime scales with the number of Monte Carlo paths.
    """

    #Create table of results
    print("\nTable of runtimes")
    print("-" * 118)
    print(
        f"{'Paths':>12} | "
        f"{'Pure Python':>12} | "
        f"{'NumPy':>10} | "
        f"{'Numba':>10} | "
        f"{'NumPy Speedup':>15} | "
        f"{'Numba Speedup':>15}"
    )
    print("-" * 118)

    py_times = []
    np_times = []
    nb_times = []
    np_speedup_ratio = []
    nb_speedup_ratio = []
    
    # Trigger Numba's JIT compilation.
    warmup = np.random.default_rng(seed).standard_normal(1000)
    _ = mc_call_numba(s0, k, r, sigma, t, warmup)


    for n_paths in path_sizes:
        # Generate one shared set of random samples for fair comparison.
        rng = np.random.default_rng(seed)
        normals_np = rng.standard_normal(n_paths)
        normals_list = normals_np.tolist()

        py_runs = []
        np_runs = []
        nb_runs = []

        for _ in range(n_repeats):
            py_ms = time_call(mc_call_pure_python,s0, k, r, sigma, t, normals_list)
            np_ms = time_call(mc_call_numpy,s0, k, r, sigma, t, normals_np)
            nb_ms = time_call(mc_call_numba,s0, k, r, sigma, t, normals_np)

            py_runs.append(py_ms)
            np_runs.append(np_ms)
            nb_runs.append(nb_ms)

        # Use the median runtime 
        py_ms = float(np.median(py_runs))
        np_ms = float(np.median(np_runs))
        nb_ms = float(np.median(nb_runs))

        numpy_speedup = py_ms / np_ms
        numba_speedup = py_ms / nb_ms

        py_times.append(py_ms)
        np_times.append(np_ms)
        nb_times.append(nb_ms)
        np_speedup_ratio.append(numpy_speedup)
        nb_speedup_ratio.append(numba_speedup)

        print(
            f"{n_paths:12,d} | "
            f"{py_ms:12.2f} | "
            f"{np_ms:10.2f} | "
            f"{nb_ms:10.2f} | "
            f"{numpy_speedup:15.2f}x | "
            f"{numba_speedup:15.2f}x"
        )

    # Runtime plot
    plt.figure(figsize=(8, 5))
    plt.plot(path_sizes, py_times, marker="o", label="Pure Python")
    plt.plot(path_sizes, np_times, marker="o", label="NumPy")
    plt.plot(path_sizes, nb_times, marker="o", label="Numba")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Monte Carlo paths")
    plt.ylabel("Median runtime (ms)")
    plt.title("Runtime Scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Speedup plot
    plt.figure(figsize=(8, 5))
    plt.plot(path_sizes, np_speedup_ratio, marker="o", label="NumPy speedup")
    plt.plot(path_sizes, nb_speedup_ratio, marker="o", label="Numba speedup")
    plt.xscale("log")
    plt.xlabel("Number of Monte Carlo paths")
    plt.ylabel("Speedup multiple ")
    plt.title("Speedup vs Pure Python")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run simulator
def main():
    compare_runtime(
        s0=100.0,
        k=100.0,
        r=0.05,
        sigma=0.20,
        t=1.0,
        path_sizes=[
            1_000,
            2_000,
            5_000,
            10_000,
            20_000,
            50_000,
            100_000,
            200_000,
            500_000,
            1_000_000,
        ],
    )


if __name__ == "__main__":
    main()