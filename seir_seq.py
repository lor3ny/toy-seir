import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class DiseaseParameters:
    """Parameters defining a specific disease"""
    name: str
    r0: float                    # Basic reproduction number
    incubation_days: float       # Days in Exposed state
    infectious_days: float       # Days in Infectious state
    fatality_rate: float         # Proportion who die (0-1)
    recovery_days: float         # Days to recover if survive
    asymptomatic_rate: float = 0.0  # Proportion infectious but no symptoms
    starting_location: str = "downtown"
    
    def get_beta(self, population_density: float = 1.0) -> float:
        """Calculate transmission rate from R0"""
        # beta = R0 / (infectious_period)
        return self.r0 / self.infectious_days
    
    def get_sigma(self) -> float:
        """Rate of progression from Exposed to Infectious"""
        return 1.0 / self.incubation_days
    
    def get_gamma(self) -> float:
        """Recovery rate"""
        return 1.0 / self.infectious_days

class SEIRSimulator:
    """
    SEIR epidemic model simulator
    S = Susceptible
    E = Exposed (infected but not yet infectious)
    I = Infectious
    R = Recovered
    D = Dead
    """
    
    def __init__(self, 
                 disease: DiseaseParameters,
                 population: int = 100000,
                 initial_infected: int = 10):
        
        self.disease = disease
        self.N = population  # Total population
        
        # Initial conditions
        self.S = population - initial_infected  # Susceptible
        self.E = 0                               # Exposed
        self.I = initial_infected                # Infectious
        self.R = 0                               # Recovered
        self.D = 0                               # Dead
        
        # Calculate rates from disease parameters
        # CORRECTED: beta must account for contact rate
        # beta = R0 * gamma (recovery rate)
        # This ensures R0 infections per infectious period
        self.gamma = disease.get_gamma()  # Recovery rate
        self.beta = disease.r0 * self.gamma / population  # Transmission rate (mass action)
        self.sigma = disease.get_sigma()  # Incubation rate
        self.mu = disease.fatality_rate   # Death rate
        
        # Storage for results
        self.history = {
            'S': [self.S],
            'E': [self.E],
            'I': [self.I],
            'R': [self.R],
            'D': [self.D],
            'day': [0]
        }
    
    def step(self, dt: float = 1.0):
        """
        Advance simulation by one time step (default 1 day)
        Uses differential equations for SEIR model
        """
        
        # Current population sizes
        S, E, I, R, D = self.S, self.E, self.I, self.R, self.D
        
        # SEIR differential equations with mass action mixing
        # New exposures: contact rate * probability of contact with infectious
        new_exposures = self.beta * S * I * dt
        
        # Exposed become infectious
        new_infectious = self.sigma * E * dt
        
        # Infectious people recover or die
        new_recovered = self.gamma * I * (1 - self.mu) * dt
        new_deaths = self.gamma * I * self.mu * dt
        
        # Update compartments
        self.S -= new_exposures
        self.E += new_exposures - new_infectious
        self.I += new_infectious - new_recovered - new_deaths
        self.R += new_recovered
        self.D += new_deaths
        
        # Ensure no negative values (numerical stability)
        self.S = max(0, self.S)
        self.E = max(0, self.E)
        self.I = max(0, self.I)
        self.R = max(0, self.R)
        self.D = max(0, self.D)
        
    def simulate(self, days: int = 30, dt: float = 0.1) -> Dict[str, List[float]]:
        """
        Run simulation for specified number of days
        dt = time step size (smaller = more accurate, default 0.1 days)
        """
        
        steps_per_day = int(1.0 / dt)
        total_steps = days * steps_per_day
        
        for step in range(total_steps):
            self.step(dt=dt)
            
            # Record state once per day
            if (step + 1) % steps_per_day == 0:
                day = (step + 1) // steps_per_day
                self.history['S'].append(self.S)
                self.history['E'].append(self.E)
                self.history['I'].append(self.I)
                self.history['R'].append(self.R)
                self.history['D'].append(self.D)
                self.history['day'].append(day)
            
        return self.history
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Calculate key statistics for scoring"""
        
        total_infected = self.history['E'][-1] + self.history['I'][-1] + self.history['R'][-1] + self.history['D'][-1]
        peak_infectious = max(self.history['I'])
        total_deaths = self.history['D'][-1]
        attack_rate = (total_infected / self.N) * 100  # Percentage infected
        
        # Estimate hospital load (assume 10% of infectious need hospitalization)
        peak_hospital_load = peak_infectious * 0.10
        
        return {
            'total_infected': total_infected,
            'attack_rate': attack_rate,
            'peak_infectious': peak_infectious,
            'peak_hospital_load': peak_hospital_load,
            'total_deaths': total_deaths,
            'case_fatality_rate': (total_deaths / total_infected * 100) if total_infected > 0 else 0
        }


def create_historical_scenarios() -> Dict[str, DiseaseParameters]:
    """Define our 5 historical disease scenarios"""
    
    scenarios = {
        'black_death': DiseaseParameters(
            name="Black Death (1347)",
            r0=2.5,
            incubation_days=2.5,
            infectious_days=5.0,
            fatality_rate=0.40,
            recovery_days=10.0,
            starting_location="port"
        ),
        
        'spanish_flu': DiseaseParameters(
            name="Spanish Flu (1918)",
            r0=2.0,
            incubation_days=1.5,
            infectious_days=7.0,
            fatality_rate=0.025,
            recovery_days=8.0,
            starting_location="military_base"
        ),
        
        'covid19': DiseaseParameters(
            name="COVID-19 Original (2020)",
            r0=2.8,
            incubation_days=5.0,
            infectious_days=10.0,
            fatality_rate=0.015,
            recovery_days=14.0,
            asymptomatic_rate=0.40,
            starting_location="airport"
        ),
        
        'measles': DiseaseParameters(
            name="Measles (unvaccinated)",
            r0=15.0,
            incubation_days=11.0,
            infectious_days=8.0,
            fatality_rate=0.002,
            recovery_days=10.0,
            starting_location="school"
        ),
        
        'ebola': DiseaseParameters(
            name="Ebola (2014)",
            r0=1.8,
            incubation_days=9.0,
            infectious_days=10.0,
            fatality_rate=0.50,
            recovery_days=14.0,
            starting_location="hospital"
        )
    }
    
    return scenarios


def plot_comparison(all_results: List[Dict], scenarios: Dict[str, DiseaseParameters], population: int = 100000, days: int = 30, initial_infected: int = 10):
    """
    Create comprehensive visualization comparing all 5 plagues
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define colors for each disease
    colors = {
        'black_death': '#8B0000',      # Dark red
        'spanish_flu': '#FF6B6B',      # Light red
        'covid19': '#4ECDC4',          # Teal
        'measles': '#FFE66D',          # Yellow
        'ebola': '#2C3E50'             # Dark blue-gray
    }
    
    # 1. INFECTIOUS OVER TIME (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    for scenario_key, result in all_results.items():
        disease_name = scenarios[scenario_key].name
        ax1.plot(result['history']['day'], 
                result['history']['I'],
                label=disease_name,
                linewidth=2.5,
                color=colors[scenario_key])
    
    ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Infectious Population', fontsize=11, fontweight='bold')
    ax1.set_title('Race: Active Infections Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, days)
    
    # 2. CUMULATIVE INFECTIONS (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    for scenario_key, result in all_results.items():
        disease_name = scenarios[scenario_key].name
        history = result['history']
        # Total infected = E + I + R + D
        cumulative = [e + i + r + d for e, i, r, d in 
                     zip(history['E'], history['I'], history['R'], history['D'])]
        ax2.plot(history['day'], 
                cumulative,
                label=disease_name,
                linewidth=2.5,
                color=colors[scenario_key])
    
    ax2.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Ever Infected', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Impact Over Time', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', frameon=True, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, days)
    
    # 3. DEATHS OVER TIME (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    for scenario_key, result in all_results.items():
        disease_name = scenarios[scenario_key].name
        ax3.plot(result['history']['day'], 
                result['history']['D'],
                label=disease_name,
                linewidth=2.5,
                color=colors[scenario_key])
    
    ax3.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Total Deaths', fontsize=11, fontweight='bold')
    ax3.set_title('Lethality: Deaths Over Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', frameon=True, fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, days)
    
    # 4. SEIR COMPARTMENTS STACKED (middle right - pick most interesting disease)
    ax4 = fig.add_subplot(gs[1, 1])
    # Let's show COVID-19 as example
    covid_result = all_results['covid19']
    days = covid_result['history']['day']
    
    ax4.fill_between(days, 0, covid_result['history']['S'], 
                     label='Susceptible', color='lightblue', alpha=0.7)
    ax4.fill_between(days, covid_result['history']['S'], 
                     [s+e for s,e in zip(covid_result['history']['S'], covid_result['history']['E'])],
                     label='Exposed', color='orange', alpha=0.7)
    ax4.fill_between(days, 
                     [s+e for s,e in zip(covid_result['history']['S'], covid_result['history']['E'])],
                     [s+e+i for s,e,i in zip(covid_result['history']['S'], 
                                            covid_result['history']['E'],
                                            covid_result['history']['I'])],
                     label='Infectious', color='red', alpha=0.7)
    ax4.fill_between(days,
                     [s+e+i for s,e,i in zip(covid_result['history']['S'], 
                                            covid_result['history']['E'],
                                            covid_result['history']['I'])],
                     [s+e+i+r for s,e,i,r in zip(covid_result['history']['S'], 
                                                 covid_result['history']['E'],
                                                 covid_result['history']['I'],
                                                 covid_result['history']['R'])],
                     label='Recovered', color='green', alpha=0.7)
    
    ax4.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Population', fontsize=11, fontweight='bold')
    ax4.set_title('Population Flow (COVID-19 Example)', fontsize=13, fontweight='bold')
    ax4.legend(loc='right', frameon=True, fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 30)
    
    # 5. FINAL SCOREBOARD (bottom, spanning both columns)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate scores
    scores_data = []
    for scenario_key, result in all_results.items():
        stats = result['stats']
        score = (stats['total_infected'] * 0.5 + 
                stats['peak_hospital_load'] * 0.3 + 
                stats['total_deaths'] * 0.2)
        scores_data.append({
            'name': scenarios[scenario_key].name,
            'score': score,
            'infected': stats['total_infected'],
            'deaths': stats['total_deaths'],
            'peak_hospital': stats['peak_hospital_load'],
            'key': scenario_key
        })
    
    # Sort by score
    scores_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Create scoreboard table
    table_data = [['Rank', 'Disease', 'Total Score', 'Infected', 'Deaths', 'Peak Hospital']]
    for rank, item in enumerate(scores_data, 1):
        table_data.append([
            f"{rank}",
            item['name'],
            f"{item['score']:,.0f}",
            f"{item['infected']:,.0f}",
            f"{item['deaths']:,.0f}",
            f"{item['peak_hospital']:,.0f}"
        ])
    
    # Draw table
    table = ax5.table(cellText=table_data, 
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.08, 0.30, 0.15, 0.15, 0.15, 0.17])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color-code ranks
    rank_colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#E8E8E8', '#F5F5F5']  # Gold, Silver, Bronze, etc.
    for i in range(1, 6):
        for j in range(6):
            cell = table[(i, j)]
            cell.set_facecolor(rank_colors[i-1])
            if j == 0:  # Rank column
                cell.set_text_props(weight='bold', fontsize=12)
    
    # Winner highlight
    winner_cell = table[(1, 1)]
    winner_cell.set_text_props(weight='bold', fontsize=11, color='darkred')
    
    # Overall title
    fig.suptitle('HISTORICAL PLAGUE COMPETITION - 30 DAY SIMULATION RESULTS', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add subtitle
    fig.text(0.5, 0.95, 'Population: 100,000 | Initial Infected: 100 | Scoring: 50% Infected + 30% Hospital + 20% Deaths',
            ha='center', fontsize=11, style='italic')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


# TEST THE MODEL
if __name__ == "__main__":
    import argparse as ap
    
    parser = ap.ArgumentParser(description="SEIR Simulator - Historical Plague Competition")
    parser.add_argument("population", help="Total population size", type=int)
    parser.add_argument("initial_infected", help="Initial number of infected individuals", type=int)
    parser.add_argument("days", help="Number of days to simulate", type=int)
    parser.add_argument("--no-viz", help="Disable visualization", action="store_true")
    parser.add_argument("--output", help="Output filename for visualization", type=str, default="plague_competition.png")
    args = parser.parse_args()
    
    # Load scenarios
    scenarios = create_historical_scenarios()
    
    print("="*70)
    print(f"HISTORICAL PLAGUE COMPETITION - {args.days} DAY SIMULATION")
    print(f"Population: {args.population:,} | Initial Infected: {args.initial_infected}")
    print("="*70)
    
    results_summary = []
    all_results = {}  # Store full history for visualization
    
    # Run all scenarios
    for scenario_key, disease in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Running: {disease.name}")
        print(f"{'='*70}")
        
        # Create simulator
        sim = SEIRSimulator(
            disease=disease, 
            population=args.population, 
            initial_infected=args.initial_infected
        )
        
        # Run simulation
        results = sim.simulate(days=args.days)
        
        # Get statistics
        stats = sim.get_summary_stats()
        
        # Store complete results for visualization
        all_results[scenario_key] = {
            'history': results,
            'stats': stats,
            'disease': disease
        }
        
        # Print results
        print(f"\nDisease Parameters:")
        print(f"  R₀: {disease.r0}")
        print(f"  Incubation Period: {disease.incubation_days} days")
        print(f"  Infectious Period: {disease.infectious_days} days")
        print(f"  Fatality Rate: {disease.fatality_rate*100:.1f}%")
        
        print(f"\n{args.days}-Day Results:")
        print(f"  Total Infected: {stats['total_infected']:,.0f} ({stats['attack_rate']:.1f}%)")
        print(f"  Peak Infectious: {stats['peak_infectious']:,.0f}")
        print(f"  Peak Hospital Load: {stats['peak_hospital_load']:,.0f} patients")
        print(f"  Total Deaths: {stats['total_deaths']:,.0f}")
        print(f"  Case Fatality Rate: {stats['case_fatality_rate']:.2f}%")
        
        # Store for final ranking
        results_summary.append({
            'name': disease.name,
            'total_infected': stats['total_infected'],
            'attack_rate': stats['attack_rate'],
            'peak_infectious': stats['peak_infectious'],
            'peak_hospital_load': stats['peak_hospital_load'],
            'total_deaths': stats['total_deaths'],
            'stats': stats
        })
    
    # Calculate scores and rank
    print("\n" + "="*70)
    print("FINAL SCOREBOARD - RANKED BY TOTAL IMPACT")
    print("="*70)
    print("\nScoring: 50% Total Infected + 30% Peak Hospital + 20% Deaths")
    print()
    
    for result in results_summary:
        # Calculate composite score
        score = (result['total_infected'] * 0.5 + 
                result['peak_hospital_load'] * 0.3 + 
                result['total_deaths'] * 0.2)
        result['score'] = score
    
    # Sort by score
    results_summary.sort(key=lambda x: x['score'], reverse=True)
    
    # Print rankings
    for rank, result in enumerate(results_summary, 1):
        print(f"{rank}. {result['name']}")
        print(f"   Score: {result['score']:,.0f} points")
        print(f"   Infected: {result['total_infected']:,.0f} ({result['attack_rate']:.1f}%)")
        print(f"   Deaths: {result['total_deaths']:,.0f}")
        print(f"   Peak Hospital: {result['peak_hospital_load']:,.0f}")
        print()
    
    print("="*70)
    print(f"WINNER: {results_summary[0]['name']}")
    print("="*70)
    
    # CREATE VISUALIZATIONS (unless disabled)
    if not args.no_viz:
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS...")
        print("="*70)
        
        # try:
        #     # Main comparison plot
        #     fig = plot_comparison(all_results, scenarios, population=args.population, days=args.days, initial_infected=args.initial_infected)
        #     plt.savefig(args.output, dpi=300, bbox_inches='tight')
        #     print(f"\n✓ Saved comparison plot: {args.output}")
        #     # Show the plot
        #     plt.show()
            
        # except Exception as e:
        #     print(f"\n⚠ Visualization failed: {e}")
        #     print("Continuing without visualization...")

        # Main comparison plot
        fig = plot_comparison(all_results, scenarios, population=args.population, days=args.days, initial_infected=args.initial_infected)
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comparison plot: {args.output}")
        # Show the plot
        plt.show()

    print("\n" + "="*70)
    print("SIMULATION COMPLETE!")
    print("="*70)