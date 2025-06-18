# CVR Valuation Model

This repository presents a robust Python-based model for valuing Contingent Value Rights (CVRs) in corporate merger scenarios.

## What Does This Do?

This tool is designed to:
* Calculate a range of potential share prices for a CVR's extension.
* Determine the risk-neutral probability of extending the option.
* Incorporate quarterly dividends within a binomial option pricing framework.
* Identify indifference points using `scipy.optimize.brentq` for precise valuation.
* Provide stochastic probability estimates via Monte Carlo simulation (relying on Brownian Motion).

Essentially, it's a specialized financial instrument valuation utility for a real-world corporate finance application.

## Key Features

* **Binomial Model Core:** Provides a solid foundation for derivative pricing.
* **Monte Carlo Simulation:** Ensures robust stochastic probability estimation.
* **Zero-Finding Algorithm:** Enhances precision for indifference point analysis.

Feel free to explore the codebase and its implementation.

---
