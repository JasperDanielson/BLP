#######################################################
# BLP Demand Estimation for Differentiated Products Markets
#######################################################

from __future__ import annotations

import logging
from typing import Final

import numpy as np
import pandas as pd
import pyblp
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

# -----------------------------------------------------------------------------
# Globals & Configuration
# -----------------------------------------------------------------------------

LOG_LEVEL: Final[int] = logging.INFO
RNG_SEED: Final[int] = 123

NUM_FIRMS: Final[int] = 4
NUM_MARKETS: Final[int] = 600
TOTAL_OBS: Final[int] = NUM_FIRMS * NUM_MARKETS

# Structural parameters used to generate the synthetic data
BETA_X: Final[float] = 1.0
BETA_SAT: Final[float] = 4.0
BETA_SAT_SD: Final[float] = 1.0
BETA_WIRE: Final[float] = 4.0
BETA_WIRE_SD: Final[float] = 1.0
ALPHA_PRICE: Final[float] = -2.0
GAMMA_CONST: Final[float] = 0.5
GAMMA_W: Final[float] = 0.25

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def configure_logging() -> None:
    """Configure root logger for clean, timestamped console output."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Generate the simulated product‑market dataset."""
    market_ids = np.repeat(np.arange(NUM_MARKETS), NUM_FIRMS)
    firm_ids = np.tile(np.arange(NUM_FIRMS), NUM_MARKETS)

    is_satellite = np.tile([1, 1, 0, 0], NUM_MARKETS)
    is_wired = np.tile([0, 0, 1, 1], NUM_MARKETS)

    product_quality = np.abs(rng.standard_normal(TOTAL_OBS))
    cost_shifter = np.abs(rng.standard_normal(TOTAL_OBS))

    # Correlated unobservables (ξ – demand shock, ω – cost shock)
    unobs = rng.multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, 0.25], [0.25, 1.0]], size=TOTAL_OBS
    )
    xi_values = unobs[:, 0]
    omega_values = unobs[:, 1] / 8.0  # scale to moderate variance

    return pd.DataFrame(
        {
            "market_ids": market_ids,
            "firm_ids": firm_ids,
            "x": product_quality,
            "satellite": is_satellite,
            "wired": is_wired,
            "w": cost_shifter,
            "xi": xi_values,
            "omega": omega_values,
        }
    )


def simulate(data: pd.DataFrame) -> tuple[pd.DataFrame, pyblp.Simulation]:
    """Run a pyblp Simulation on the synthetic data."""
    integration = pyblp.Integration("product", 9)

    x1 = pyblp.Formulation("0 + x + satellite + wired + prices")
    x2 = pyblp.Formulation("0 + satellite + wired")
    x3 = pyblp.Formulation("1 + w")

    simulation = pyblp.Simulation(
        product_formulations=(x1, x2, x3),
        product_data=data,
        beta=[BETA_X, BETA_SAT, BETA_WIRE, ALPHA_PRICE],
        sigma=[[BETA_SAT_SD, 0.0], [0.0, BETA_WIRE_SD]],
        gamma=[GAMMA_CONST, GAMMA_W],
        costs_type="log",
        xi=data["xi"].to_numpy(),
        omega=data["omega"].to_numpy(),
        seed=RNG_SEED,
        integration=integration,
    )

    transformed = pd.DataFrame(
        pyblp.data_to_dict(simulation.replace_endogenous().product_data)
    )
    return transformed, simulation


def estimate_ols(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Estimate demand parameters via simple OLS."""
    shares = df.groupby("market_ids")["shares"].transform("sum")
    y = np.log(df["shares"] / (1 - shares))
    X = df[["x", "satellite", "wired", "prices"]]
    return sm.OLS(endog=y, exog=X).fit()


def estimate_2sls(df: pd.DataFrame) -> IV2SLS:
    """Estimate demand parameters via two‑stage least squares."""
    shares = df.groupby("market_ids")["shares"].transform("sum")
    y = np.log(df["shares"] / (1 - shares))

    exog = df[["x", "satellite", "wired"]]
    endog = df["prices"]
    instruments = df[["w"]]

    return IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instruments).fit()


def estimate_blp(
    df: pd.DataFrame, integration: pyblp.Integration
) -> pyblp.results.ProblemResults:
    """Estimate the random‑coefficients logit model (BLP)."""
    df = df.copy()
    blp_instr = pyblp.build_blp_instruments(
        formulation=pyblp.Formulation("0 + x"), product_data=df
    )
    df["demand_iv"] = blp_instr[:, 1]

    problem = pyblp.Problem(
        product_formulations=(
            pyblp.Formulation("0 + x + satellite + wired + prices"),
            pyblp.Formulation("0 + satellite + wired"),
        ),
        product_data=df,
        integration=integration,
        add_exogenous=True,
    )
    # Use true sigma as a starting point to speed convergence
    return problem.solve(sigma=[[BETA_SAT_SD, 0.0], [0.0, BETA_WIRE_SD]])

# -----------------------------------------------------------------------------
# Main script logic
# -----------------------------------------------------------------------------


def main() -> None:
    configure_logging()
    rng = np.random.default_rng(RNG_SEED)

    logging.info("Building synthetic product‑market data …")
    base_df = build_dataframe(rng)

    logging.info("Running BLP simulation …")
    prod_df, sim = simulate(base_df)

    logging.info("Estimating OLS benchmark …")
    ols_res = estimate_ols(prod_df)
    logging.info("\n%s", ols_res.summary())

    logging.info("Estimating 2SLS benchmark …")
    iv_res = estimate_2sls(prod_df)
    logging.info("\n%s", iv_res.summary)

    logging.info("Estimating random‑coefficients BLP model …")
    blp_res = estimate_blp(prod_df, sim.integration)
    logging.info("\n%s", blp_res)


if __name__ == "__main__":
    main()
