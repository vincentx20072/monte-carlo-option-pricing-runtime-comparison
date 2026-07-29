# Benchmarking Monte Carlo European Option Pricing in Python

A performance comparison of Pure Python, NumPy, and Numba implementations of a Monte Carlo European call option pricer, demonstrating the impact of vectorisation and just-in-time (JIT) compilation on runtime.

The project benchmarks three implementations of the same Monte Carlo simulation using identical random samples, allowing for a fair comparison of execution time as the number of simulation paths increases.

---

## Features

- European call option pricing using Monte Carlo simulation
- Three implementations:
  - Pure Python
  - NumPy vectorisation
  - Numba JIT compilation
- Runtime benchmarking across multiple Monte Carlo path sizes
- Runtime scaling visualisation
- Speedup comparison multiples relative to the Pure Python implementation
- Shared random samples for consistent benchmarking

---

## Mathematical Model

The terminal stock price is simulated using the closed-form solution of Geometric Brownian Motion:

```text
S_T = S_0 × exp((r − 0.5σ²)T + σ√T Z)
```

where:

- `S_0` = initial stock price
- `K` = strike price
- `r` = risk-free interest rate
- `σ` = volatility
- `T` = time to maturity (years)
- `Z ~ N(0,1)` = standard normally distributed random variable

The European call option payoff is given by:

```text
Payoff = max(S_T − K, 0)
```

The option price is then estimated as the discounted average payoff across all N simulated paths:

```text
Option Price = exp(−rT) × (1/N) Σ max(S_T^(i) − K, 0)
```

---

## Requirements

- Python 3.9+
- NumPy
- Numba
- Matplotlib

Install the required packages:

```bash
pip install numpy numba matplotlib
```

---

## Running the Benchmark

Run the script directly:

```bash
python main.py
```

The benchmark will:

1. Warm up the Numba JIT compiler.
2. Execute each implementation across multiple simulation sizes.
3. Print a runtime comparison table.
4. Plot runtime scaling.
5. Plot speedup relative to the Pure Python implementation.

---

## Example Parameters

The benchmark uses:

| Parameter | Value |
|----------|------:|
| Initial stock price (\(S_0\)) | 100 |
| Strike price (\(K\)) | 100 |
| Risk-free rate (\(r\)) | 5% |
| Volatility (\(\sigma\)) | 20% |
| Time to maturity (\(T\)) | 1 year |

Simulation sizes range from **1,000** to **1,000,000** Monte Carlo paths.

---

## Project Structure

```
.
├── main.py
└── README.md
```

---

## Implementation Overview

### Pure Python

Uses a standard Python loop over every Monte Carlo path.

- Simple and readable
- Serves as the baseline implementation
- Slowest execution

### NumPy

Uses vectorised array operations instead of explicit Python loops.

- Eliminates Python loop overhead
- Significantly faster for numerical workloads
- Efficient use of NumPy's optimised C backend

### Numba

Uses Numba's Just-In-Time (JIT) compiler with parallel execution.

- Compiles numerical code to machine code
- Parallelises the Monte Carlo loop
- Typically provides the fastest runtime for large simulations

---

## Notes

- All implementations use the same set of normally distributed random numbers for a fair performance comparison.
- Numba is warmed up before benchmarking so compilation time is excluded from the reported results.
- Median runtime over multiple runs is reported to reduce measurement noise.

