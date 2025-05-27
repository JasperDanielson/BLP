#######################################################
# BLP Merger Simulation and Welfare Analysis
#######################################################

import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import linearmodels as lm
import pyblp

# Configure environment
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore', category=FutureWarning)
plt.style.use('seaborn-v0_8')
RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)


@dataclass
class MergerScenario:
    """Data class to store merger scenario parameters and results."""
    name: str
    merging_firms: List[int]
    cost_reduction: float = 0.0
    description: str = ""


@dataclass
class WelfareResults:
    """Data class to store welfare analysis results."""
    consumer_surplus: float
    producer_profits: float
    total_welfare: float
    price_changes: np.ndarray
    share_changes: np.ndarray


class BLPMergerSimulator:
    """
    A comprehensive merger simulation framework using BLP demand estimation.
    
    This class extends the BLP demand estimation to analyze the competitive
    effects of horizontal mergers, including price effects, market share
    redistribution, and welfare implications.
    """
    
    def __init__(self):
        """Initialize the merger simulator with market parameters."""
        # Market structure parameters
        self.num_firms = 4
        self.num_markets = 600
        self.total_obs = self.num_firms * self.num_markets
        
        # True demand parameters (for validation)
        self.true_params = {
            'beta_x': 1.0,
            'beta_sat': 4.0,
            'beta_wire': 4.0,
            'beta_sat_sd': 1.0,
            'beta_wire_sd': 1.0,
            'alpha_price': -2.0,
            'gamma_const': 0.5,
            'gamma_w': 0.25
        }
        
        # Initialize containers for results
        self.market_data: Optional[pd.DataFrame] = None
        self.blp_results: Optional[pyblp.ProblemResults] = None
        self.baseline_metrics: Optional[Dict] = None
        
        # Random number generator
        self.rng = np.random.default_rng(seed=RANDOM_SEED)
        
        logging.info("Initialized BLP merger simulator for antitrust analysis")
    
    def _generate_market_data(self) -> pd.DataFrame:
        """Generate simulated market equilibrium data for merger analysis."""
        logging.info("Generating market simulation data...")
        
        # Market and firm identifiers
        market_ids = np.repeat(np.arange(self.num_markets), self.num_firms)
        firm_ids = np.tile(np.arange(self.num_firms), self.num_markets)
        
        # Product characteristics
        is_satellite = np.tile([1, 1, 0, 0], self.num_markets)
        is_wired = np.tile([0, 0, 1, 1], self.num_markets)
        
        # Observable characteristics
        product_quality = np.abs(self.rng.standard_normal(self.total_obs))
        cost_shifter = np.abs(self.rng.standard_normal(self.total_obs))
        
        # Correlated unobservables
        unobs_cov = np.array([[1.0, 0.25], [0.25, 1.0]])
        unobs_components = self.rng.multivariate_normal(
            mean=[0, 0], cov=unobs_cov, size=self.total_obs
        )
        
        # Create base DataFrame
        data = pd.DataFrame({
            'market_ids': market_ids,
            'firm_ids': firm_ids,
            'x': product_quality,
            'sat': is_satellite,
            'wired': is_wired,
            'w': cost_shifter,
            'xi': unobs_components[:, 0],
            'omega': unobs_components[:, 1],
            'adjusted_omega': unobs_components[:, 1] / 8
        })
        
        return data
    
    def _run_blp_estimation(self, data: pd.DataFrame) -> pyblp.ProblemResults:
        """Execute full BLP estimation with optimal instruments."""
        logging.info("Running BLP estimation for merger simulation...")
        
        # Define formulations
        demand_formulation = pyblp.Formulation("0 + x + sat + wired + prices")
        random_coeff_formulation = pyblp.Formulation("0 + sat + wired")
        supply_formulation = pyblp.Formulation("1 + w")
        
        # Integration setup
        integration = pyblp.Integration("product", size=9)
        
        # Simulate equilibrium
        simulation = pyblp.Simulation(
            product_formulations=(demand_formulation, random_coeff_formulation, supply_formulation),
            product_data=data,
            beta=[self.true_params['beta_x'], self.true_params['beta_sat'], 
                  self.true_params['beta_wire'], self.true_params['alpha_price']],
            sigma=np.array([[self.true_params['beta_sat_sd'], 0], 
                           [0, self.true_params['beta_wire_sd']]]),
            gamma=[self.true_params['gamma_const'], self.true_params['gamma_w']],
            costs_type="log",
            xi=data['xi'].values,
            omega=data['adjusted_omega'].values,
            seed=RANDOM_SEED,
            integration=integration
        )
        
        # Generate equilibrium data
        sim_result = simulation.replace_endogenous()
        market_data = pd.DataFrame(pyblp.data_to_dict(sim_result.product_data))
        
        # Build comprehensive instrument set
        blp_demand_iv = pyblp.build_blp_instruments(
            formulation=pyblp.Formulation("0 + x"), product_data=market_data
        )
        blp_diff_demand_iv = pyblp.build_differentiation_instruments(
            formulation=pyblp.Formulation("0 + x"), product_data=market_data
        )
        blp_supply_iv = pyblp.build_blp_instruments(
            formulation=pyblp.Formulation("0 + w"), product_data=market_data
        )
        blp_diff_supply_iv = pyblp.build_differentiation_instruments(
            formulation=pyblp.Formulation("0 + w"), product_data=market_data
        )
        
        # Add instruments to data
        market_data['demand_instruments0'] = blp_demand_iv[:, 1]
        market_data['demand_instruments1'] = market_data['w']
        market_data['demand_instruments2'] = blp_diff_demand_iv[:, 1]
        market_data['supply_instruments0'] = blp_supply_iv[:, 1]
        market_data['supply_instruments1'] = blp_diff_supply_iv[:, 1]
        
        # Estimate demand model
        demand_problem = pyblp.Problem(
            product_formulations=(demand_formulation, random_coeff_formulation),
            product_data=market_data,
            integration=integration,
            add_exogenous=True
        )
        
        demand_results = demand_problem.solve(sigma=simulation.sigma)
        
        # Compute optimal instruments for efficiency
        optimal_iv_problem = demand_results.compute_optimal_instruments().to_problem()
        
        # Final estimation with optimal instruments
        final_results = optimal_iv_problem.solve(
            sigma=simulation.sigma,
            beta=[None, None, None, -1],
            optimization=pyblp.Optimization("bfgs", {"gtol": 1e-4}),
            iteration=pyblp.Iteration("squarem", {"atol": 1e-13})
        )
        
        self.market_data = market_data
        logging.info("BLP estimation completed successfully")
        return final_results
    
    def _compute_baseline_metrics(self) -> Dict:
        """Compute baseline market metrics before merger."""
        logging.info("Computing baseline market metrics...")
        
        baseline_costs = self.blp_results.compute_costs()
        baseline_cs = self.blp_results.compute_consumer_surpluses()
        baseline_profits = self.blp_results.compute_profits()
        
        # Compute elasticities
        elasticities = self.blp_results.compute_elasticities()
        own_price_elasticities = self._extract_own_price_elasticities(elasticities)
        
        return {
            'costs': baseline_costs,
            'consumer_surplus': baseline_cs.mean(),
            'total_profits': baseline_profits.reshape((self.num_markets, self.num_firms)).mean(axis=0).sum(),
            'firm_profits': baseline_profits.reshape((self.num_markets, self.num_firms)).mean(axis=0),
            'prices': self.market_data['prices'].values,
            'shares': self.market_data['shares'].values,
            'own_elasticities': own_price_elasticities
        }
    
    def _extract_own_price_elasticities(self, elasticity_matrix: np.ndarray) -> np.ndarray:
        """Extract own-price elasticities from full elasticity matrix."""
        reshaped = elasticity_matrix.reshape((self.num_markets, self.num_firms, self.num_firms))
        own_elasticities = np.diagonal(reshaped, axis1=1, axis2=2)
        return own_elasticities.mean(axis=0)
    
    def _validate_elasticities(self) -> None:
        """Validate estimated elasticities against theoretical values."""
        logging.info("Validating elasticity estimates...")
        
        # Compute theoretical own-price elasticities
        alpha = self.true_params['alpha_price']
        prices = self.market_data['prices'].values.reshape((self.num_markets, self.num_firms))
        shares = self.market_data['shares'].values.reshape((self.num_markets, self.num_firms))
        
        theoretical_elasticities = []
        for j in range(self.num_firms):
            firm_elasticity = (alpha * prices[:, j] * (1 - shares[:, j])).mean()
            theoretical_elasticities.append(firm_elasticity)
            
        estimated = self.baseline_metrics['own_elasticities']
        theoretical = np.array(theoretical_elasticities)
        
        print("\nElasticity Validation:")
        print("-" * 40)
        for i in range(self.num_firms):
            print(f"Firm {i+1} - Estimated: {estimated[i]:.3f}, Theoretical: {theoretical[i]:.3f}")
    
    def simulate_merger(self, scenario: MergerScenario) -> WelfareResults:
        """
        Simulate the effects of a horizontal merger.
        
        Args:
            scenario: MergerScenario object with merger details
            
        Returns:
            WelfareResults object with post-merger outcomes
        """
        logging.info(f"Simulating merger: {scenario.name}")
        
        # Create merger firm mapping
        merger_ids = self.market_data['firm_ids'].copy()
        target_firm = scenario.merging_firms[0]  # Acquiring firm
        
        for firm in scenario.merging_firms[1:]:
            merger_ids = merger_ids.replace(firm, target_firm)
        
        # Apply cost synergies if specified
        post_merger_costs = self.baseline_metrics['costs'].copy()
        if scenario.cost_reduction > 0:
            merged_firm_mask = merger_ids == target_firm
            post_merger_costs[merged_firm_mask] *= (1 - scenario.cost_reduction)
        
        # Compute post-merger equilibrium
        post_merger_prices = self.blp_results.compute_prices(
            firm_ids=merger_ids, costs=post_merger_costs
        )
        post_merger_shares = self.blp_results.compute_shares(post_merger_prices)
        post_merger_cs = self.blp_results.compute_consumer_surpluses(post_merger_prices)
        post_merger_profits = self.blp_results.compute_profits(post_merger_prices)
        
        # Calculate changes
        price_changes = (post_merger_prices.reshape((self.num_markets, self.num_firms)) - 
                        self.baseline_metrics['prices'].reshape((self.num_markets, self.num_firms)))
        share_changes = (post_merger_shares.reshape((self.num_markets, self.num_firms)) - 
                        self.baseline_metrics['shares'].reshape((self.num_markets, self.num_firms)))
        
        return WelfareResults(
            consumer_surplus=post_merger_cs.mean(),
            producer_profits=post_merger_profits.reshape((self.num_markets, self.num_firms)).mean(axis=0).sum(),
            total_welfare=post_merger_cs.mean() + post_merger_profits.reshape((self.num_markets, self.num_firms)).mean(axis=0).sum(),
            price_changes=price_changes.mean(axis=0),
            share_changes=share_changes.mean(axis=0)
        )
    
    def run_comprehensive_analysis(self) -> Dict:
        """Execute complete merger simulation analysis."""
        logging.info("Starting comprehensive merger simulation analysis")
        
        # Step 1: Generate data and estimate model
        data = self._generate_market_data()
        self.blp_results = self._run_blp_estimation(data)
        self.baseline_metrics = self._compute_baseline_metrics()
        
        # Step 2: Validate model
        self._validate_elasticities()
        
        # Step 3: Define merger scenarios
        scenarios = [
            MergerScenario(
                name="Merger 1: Firms 1 & 2",
                merging_firms=[0, 1],
                description="Horizontal merger between satellite providers"
            ),
            MergerScenario(
                name="Merger 2: Firms 1 & 3", 
                merging_firms=[0, 2],
                description="Cross-technology merger (satellite + wired)"
            ),
            MergerScenario(
                name="Merger with Synergies",
                merging_firms=[0, 1],
                cost_reduction=0.15,
                description="Merger with 15% cost reduction"
            )
        ]
        
        # Step 4: Simulate all scenarios
        results = {'baseline': self.baseline_metrics}
        for scenario in scenarios:
            results[scenario.name] = self.simulate_merger(scenario)
        
        # Step 5: Generate comprehensive output
        self._display_results(results, scenarios)
        
        return results
    
    def _display_results(self, results: Dict, scenarios: List[MergerScenario]) -> None:
        """Display comprehensive merger analysis results."""
        
        print("\n" + "="*70)
        print("COMPREHENSIVE MERGER SIMULATION ANALYSIS")
        print("="*70)
        
        # Model validation
        estimated_params = {
            'alpha': self.blp_results.beta[3, 0],
            'beta_x': self.blp_results.beta[0, 0], 
            'beta_sat': self.blp_results.beta[1, 0],
            'beta_wire': self.blp_results.beta[2, 0],
            'sigma_sat': self.blp_results.sigma[0, 0],
            'sigma_wire': self.blp_results.sigma[1, 1]
        }
        
        print("\n1. MODEL PARAMETER ESTIMATES")
        print("-" * 50)
        for param, value in estimated_params.items():
            true_val = list(self.true_params.values())[list(estimated_params.keys()).index(param)]
            print(f"{param:12}: {value:8.4f} (True: {true_val:6.3f})")
        
        # Market structure
        print(f"\n2. MARKET STRUCTURE")
        print("-" * 50)
        print(f"Markets: {self.num_markets}")
        print(f"Firms per market: {self.num_firms}")
        print(f"Average price: ${self.market_data['prices'].mean():.2f}")
        print(f"Average market share: {self.market_data['shares'].mean():.4f}")
        
        # Welfare analysis table
        print(f"\n3. WELFARE IMPACT ANALYSIS")
        print("-" * 70)
        print(f"{'Scenario':<25} {'Consumer Surplus':<15} {'Profits':<12} {'Total Welfare':<12}")
        print("-" * 70)
        
        baseline = results['baseline']
        baseline_welfare = baseline['consumer_surplus'] + baseline['total_profits']
        print(f"{'Baseline':<25} {baseline['consumer_surplus']:>12.4f} {baseline['total_profits']:>9.4f} {baseline_welfare:>9.4f}")
        
        for scenario in scenarios:
            welfare_result = results[scenario.name]
            cs_change = welfare_result.consumer_surplus - baseline['consumer_surplus']
            profit_change = welfare_result.producer_profits - baseline['total_profits']
            welfare_change = welfare_result.total_welfare - baseline_welfare
            
            print(f"{scenario.name:<25} {welfare_result.consumer_surplus:>12.4f} {welfare_result.producer_profits:>9.4f} {welfare_result.total_welfare:>9.4f}")
            print(f"{'  (Change)':<25} {cs_change:>12.4f} {profit_change:>9.4f} {welfare_change:>9.4f}")
        
        # Price effects
        print(f"\n4. AVERAGE PRICE EFFECTS BY FIRM")
        print("-" * 70)
        print(f"{'Scenario':<25} {'Firm 1':<10} {'Firm 2':<10} {'Firm 3':<10} {'Firm 4':<10}")
        print("-" * 70)
        
        baseline_prices = self.baseline_metrics['prices'].reshape((self.num_markets, self.num_firms)).mean(axis=0)
        print(f"{'Baseline Prices':<25}", end="")
        for price in baseline_prices:
            print(f"{price:>9.2f}", end=" ")
        print()
        
        for scenario in scenarios:
            welfare_result = results[scenario.name]
            print(f"{scenario.name:<25}", end="")
            for change in welfare_result.price_changes:
                print(f"{change:>+9.2f}", end=" ")
            print()
    
    def create_visualization(self, results: Dict) -> None:
        """Create comprehensive visualization of merger effects."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Merger Simulation Analysis', fontsize=16, fontweight='bold')
        
        # Price changes
        scenarios = [k for k in results.keys() if k != 'baseline']
        price_changes = np.array([results[s].price_changes for s in scenarios])
        
        axes[0, 0].bar(range(len(scenarios)), price_changes.mean(axis=1))
        axes[0, 0].set_title('Average Price Change by Merger Scenario')
        axes[0, 0].set_xticks(range(len(scenarios)))
        axes[0, 0].set_xticklabels([s.replace('Merger ', 'M') for s in scenarios], rotation=45)
        axes[0, 0].set_ylabel('Price Change ($)')
        
        # Consumer surplus
        cs_baseline = results['baseline']['consumer_surplus']
        cs_changes = [results[s].consumer_surplus - cs_baseline for s in scenarios]
        
        axes[0, 1].bar(range(len(scenarios)), cs_changes)
        axes[0, 1].set_title('Consumer Surplus Impact')
        axes[0, 1].set_xticks(range(len(scenarios)))
        axes[0, 1].set_xticklabels([s.replace('Merger ', 'M') for s in scenarios], rotation=45)
        axes[0, 1].set_ylabel('CS Change')
        
        # Welfare decomposition
        profit_baseline = results['baseline']['total_profits']
        profit_changes = [results[s].producer_profits - profit_baseline for s in scenarios]
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, cs_changes, width, label='Consumer Surplus')
        axes[1, 0].bar(x + width/2, profit_changes, width, label='Producer Profits')
        axes[1, 0].set_title('Welfare Decomposition')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([s.replace('Merger ', 'M') for s in scenarios], rotation=45)
        axes[1, 0].legend()
        
        # Market concentration (HHI)
        baseline_shares = self.baseline_metrics['shares'].reshape((self.num_markets, self.num_firms))
        baseline_hhi = (baseline_shares**2).sum(axis=1).mean() * 10000
        
        axes[1, 1].text(0.5, 0.7, f'Baseline HHI: {baseline_hhi:.0f}', 
                       transform=axes[1, 1].transAxes, fontsize=14, ha='center')
        axes[1, 1].text(0.5, 0.5, 'Post-merger HHI analysis\nwould require additional\nmarket share calculations', 
                       transform=axes[1, 1].transAxes, fontsize=12, ha='center')
        axes[1, 1].set_title('Market Concentration')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function for merger simulation analysis."""
    try:
        # Initialize and run comprehensive analysis
        simulator = BLPMergerSimulator()
        results = simulator.run_comprehensive_analysis()
        
        # Create visualizations
        # simulator.create_visualization(results)
        
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()