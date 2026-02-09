import numpy as np
import matplotlib.pyplot as plt
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
        self.beta = disease.get_beta()   # Transmission rate
        self.sigma = disease.get_sigma() # Incubation rate
        self.gamma = disease.get_gamma() # Recovery rate
        self.mu = disease.fatality_rate  # Death rate
        
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
        N_alive = self.N - D  # Only living people can transmit/catch
        
        # SEIR differential equations
        # New exposures: susceptible people meeting infectious people
        new_exposures = (self.beta * S * I / N_alive) * dt if N_alive > 0 else 0
        
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
        
    def simulate(self, days: int = 30) -> Dict[str, List[float]]:
        """Run simulation for specified number of days"""
        
        for day in range(1, days + 1):
            self.step(dt=1.0)
            
            # Record state
            self.history['S'].append(self.S)
            self.history['E'].append(self.E)
            self.history['I'].append(self.I)
            self.history['R'].append(self.R)
            self.history['D'].append(self.D)
            self.history['day'].append(day)
            
        return self.history
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Calculate key statistics for scoring"""
        
        total_infected = self.history['I'][-1] + self.history['R'][-1] + self.history['D'][-1]
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


# TEST THE MODEL
if __name__ == "__main__":
    # Load scenarios
    scenarios = create_historical_scenarios()
    
    # Test with COVID-19
    print("Testing COVID-19 scenario...")
    covid_disease = scenarios['covid19']
    sim = SEIRSimulator(disease=covid_disease, population=100000, initial_infected=10)
    
    # Run 30-day simulation
    results = sim.simulate(days=30)
    
    # Print summary
    stats = sim.get_summary_stats()
    print(f"\n{covid_disease.name} - 30 Day Results:")
    print(f"  Total Infected: {stats['total_infected']:,.0f} ({stats['attack_rate']:.1f}%)")
    print(f"  Peak Infectious: {stats['peak_infectious']:,.0f}")
    print(f"  Peak Hospital Load: {stats['peak_hospital_load']:,.0f}")
    print(f"  Total Deaths: {stats['total_deaths']:,.0f}")
    print(f"  Case Fatality Rate: {stats['case_fatality_rate']:.2f}%")
