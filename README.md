# BLP Demand Estimation and Merger Simulation

## Overview

This repository provides a comprehensive implementation of the Berry-Levinsohn-Pakes (BLP) demand estimation methodology alongside a merger simulation framework. It extends traditional demand estimation to evaluate the competitive and welfare effects of horizontal mergers within differentiated product markets. The analysis is particularly relevant for antitrust reviews and economic forecasting in competitive environments.

## Motivation

Quantitative analysts in finance, antitrust authorities, and economic researchers often employ structural econometric models to understand market behavior and forecast competitive outcomes. Implementing BLP demand estimation and merger simulations demonstrates a strong ability in econometric modeling, computational economics, and data-driven policy analysis, making this project highly relevant to quantitative trading and economic research roles.

## Project Components

The repository contains two primary scripts:

1. **Demand Estimation (`Demand_Estimation.py`)**:
   - Implements the BLP demand estimation framework.
   - Uses synthetic market data generation for robust estimation.
   - Estimates parameters via Ordinary Least Squares (OLS), Two-Stage Least Squares (2SLS), and Random Coefficient Logit (BLP).

2. **Merger Simulation (`Merger_Simulation.py`)**:
   - Extends the BLP demand estimation to simulate post-merger market outcomes.
   - Analyzes consumer surplus, producer profits, total welfare, and price and share effects.
   - Evaluates mergers under varying assumptions, including potential cost efficiencies.

## Methodologies Used

- **Berry-Levinsohn-Pakes (BLP) Demand Estimation**: Estimates consumer preferences and price sensitivities in differentiated product markets.
- **Instrumental Variables (IV) Estimation**: Corrects for endogeneity bias using IV methods, including the construction of optimal instruments.
- **Structural Merger Simulation**: Predicts post-merger equilibrium outcomes using estimated demand parameters.
- **Consumer and Welfare Analysis**: Evaluates the impact of mergers on consumer surplus, firm profits, and overall welfare.

## Implementation Highlights

- **Synthetic Data Generation**: Generates robust market simulations to ensure reliable parameter estimates.
- **Optimal Instrumentation**: Enhances estimation accuracy through advanced instrument construction techniques.
- **Elasticity Validation**: Validates estimated elasticities against theoretical benchmarks to ensure model robustness.
- **Visualization and Analysis**: Provides comprehensive tabular and graphical analyses to illustrate the impacts of mergers clearly.

## Key Results

The implementation produces detailed insights into:

- Estimated demand parameters (price elasticities, consumer valuations).
- Impact of mergers on prices, market shares, and market concentration.
- Welfare outcomes including changes in consumer surplus and firm profitability.

Example scenario outcomes include:
- **Baseline**: Establishes initial equilibrium metrics.
- **Horizontal Mergers**: Quantifies the competitive effects of mergers among firms.
- **Efficiency Gains**: Evaluates scenarios with potential cost synergies and their implications for market competition and consumer welfare.

## Repository Structure
```
│
├── Demand_Estimation.py         # BLP demand estimation implementation
├── Merger_Simulation.py         # Merger simulation extension
└── README.md                    # Project overview and documentation
```

## Prerequisites
- Python 3.8+
- Required Python packages:
  - numpy, pandas
  - pyblp
  - statsmodels, linearmodels
  - matplotlib, seaborn

Install packages via:
```bash
pip install numpy pandas pyblp statsmodels linearmodels matplotlib seaborn
```

## Relevance to Quantitative Roles
This project highlights:
- Advanced econometric modeling and estimation techniques.
- Rigorous application of instrumental variable methods.
- Skill in structural economic analysis and market forecasting.
- Clear presentation of results suitable for policy analysis and financial decision-making.

## Contact
For inquiries, discussions, or further collaboration:
- Email: j.danielson@mail.utoronto.ca
- LinkedIn: https://www.linkedin.com/in/jasperdanielson/

---

Developed by Jasper Danielson, MA Economics (Finance), University of Toronto
