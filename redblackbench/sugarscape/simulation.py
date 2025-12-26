import random
from typing import List, Dict, Any
import numpy as np

from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.environment import SugarEnvironment
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.experiment import ExperimentLogger, MetricsCalculator
from redblackbench.sugarscape.trade import TradeSystem

class SugarSimulation:
    """Main simulation controller for Sugarscape."""
    
    def __init__(self, config: SugarscapeConfig = None, agent_factory=None, experiment_name: str = "baseline"):
        self.config = config or SugarscapeConfig()
        self.env = SugarEnvironment(self.config)
        self.trade_system = TradeSystem(self.env) if self.config.enable_trade else None
        
        self.agents: List[SugarAgent] = []
        self.tick = 0
        self.next_agent_id = 1
        self.agent_factory = agent_factory or self._default_agent_factory
        
        # Experiment logging
        self.logger = ExperimentLogger(experiment_type=experiment_name, config=self.config)
        
        # Random number generator
        self.rng = random.Random(self.config.seed)
        
        self._init_population()
        
    def _default_agent_factory(self, **kwargs) -> SugarAgent:
        return SugarAgent(**kwargs)
            
    def _init_population(self):
        """Create initial population."""
        for _ in range(self.config.initial_population):
            self._create_agent()
            
    def _create_agent(self):
        """Create and place a new agent with random attributes."""
        # Random attributes
        w0 = self.rng.randint(*self.config.initial_wealth_range)
        m = self.rng.randint(*self.config.metabolism_range)
        v = self.rng.randint(*self.config.vision_range)
        max_age = self.rng.randint(*self.config.max_age_range)
        
        # Spice attributes (if enabled)
        spice = 0
        m_spice = 0
        if self.config.enable_spice:
            spice = self.rng.randint(*self.config.initial_spice_range)
            m_spice = self.rng.randint(*self.config.metabolism_spice_range)
        
        # Determine Persona
        persona = "A" # default
        if self.config.enable_personas:
            # Simple weighted choice
            dist = self.config.persona_distribution
            keys = list(dist.keys()) # A, B, C, D
            probs = [dist[k] for k in keys]
            persona = self.rng.choices(keys, weights=probs, k=1)[0]
        
        # Random position
        pos = self.env.get_random_unoccupied_pos(self.rng)
        
        agent = self.agent_factory(
            agent_id=self.next_agent_id,
            pos=pos,
            vision=v,
            metabolism=m,
            max_age=max_age,
            wealth=w0,
            age=0,
            spice=spice,
            metabolism_spice=m_spice
        )
        agent.persona = persona
        self.next_agent_id += 1
        
        self.agents.append(agent)
        self.env.add_agent(agent)
        
    def step(self):
        """Execute one simulation tick."""
        self.tick += 1
        
        # 1. Environment Growback
        self.env.growback()
        
        # 2. Agent updates (Move and Harvest)
        # Shuffle order
        self.rng.shuffle(self.agents)
        
        dead_agents = []
        
        for agent in self.agents:
            agent.step(self.env)
            if not agent.alive:
                dead_agents.append(agent)
                
        # 3. Trade Phase (New)
        if self.config.enable_trade and self.trade_system:
            # Only alive agents trade
            live_agents = [a for a in self.agents if a.alive]
            self.trade_system.execute_trade_round(live_agents)
                
        # 4. Handle deaths and replacement
        # Note: step() marks dead, but we process removal here
        # We need to re-check aliveness because trade might technically (though unlikely) affect survival if implemented that way
        # Actually standard model: trade happens before metabolism. 
        # But our agent.step() does Move -> Harvest -> Metabolize -> Die.
        # So if we trade AFTER step(), agents might have already died from metabolism.
        # Ideally: Move -> Harvest -> Trade -> Metabolize.
        # Current impl: agent.step() does it all. 
        # For simplicity in this iteration, we trade after metabolism (survivors trade).
        # OR we should split agent.step(). 
        # Given constraints, let's keep it simple: Survivors trade, then wait for next tick to metabolize again.
        
        for agent in dead_agents:
            if agent in self.agents:
                self.agents.remove(agent)
                self.env.remove_agent(agent)
                # Replacement Rule: Constant population
                self._create_agent()
            
        # Logging
        if self.tick % 10 == 0:
            stats = self.get_stats()
            self.logger.log_step(stats)
            
    def run(self, steps: int = None):
        """Run for a number of steps."""
        # Initial snapshot
        self.save_snapshot(filename="initial_state.json")
        
        limit = steps or self.config.max_ticks
        for _ in range(limit):
            self.step()
            
        # Final snapshot
        self.save_snapshot(filename="final_state.json")
            
    def save_snapshot(self, filename: str):
        """Save current simulation state."""
        data = {
            "tick": self.tick,
            "agents": [
                {
                    "id": a.agent_id,
                    "pos": a.pos,
                    "wealth": a.wealth,
                    "spice": a.spice,
                    "age": a.age,
                    "persona": a.persona,
                    "vision": a.vision,
                    "metabolism": a.metabolism,
                    "metabolism_spice": a.metabolism_spice,
                    "metrics": getattr(a, 'metrics', {})
                }
                for a in self.agents
            ],
            "sugar_map": self.env.sugar_amount.tolist(),
            "spice_map": self.env.spice_amount.tolist(),
            "sugar_capacity": self.env.sugar_capacity.tolist()
        }
        self.logger.save_snapshot(data, filename)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        wealths = [a.wealth for a in self.agents]
        spices = [a.spice for a in self.agents]
        ages = [a.age for a in self.agents]
        positions = [a.pos for a in self.agents]
        
        # Calculate Moran's I
        moran = MetricsCalculator.calculate_moran_i(
            self.config.width, self.config.height, positions, wealths
        )
        
        # Calculate Mobility
        mobility = MetricsCalculator.calculate_mobility_stats(self.agents)
        
        return {
            "tick": self.tick,
            "population": len(self.agents),
            "mean_wealth": np.mean(wealths) if wealths else 0,
            "mean_spice": np.mean(spices) if spices else 0,
            "max_wealth": np.max(wealths) if wealths else 0,
            "min_wealth": np.min(wealths) if wealths else 0,
            "gini": self._gini_coefficient(wealths) if wealths else 0,
            "mean_age": np.mean(ages) if ages else 0,
            "moran_i": moran,
            "avg_displacement": mobility["avg_displacement"],
            "avg_exploration": mobility["avg_exploration"]
        }
        
    def _gini_coefficient(self, values):
        """Calculate Gini coefficient for wealth inequality."""
        if not values: return 0
        sorted_vals = sorted(values)
        n = len(values)
        total = sum(sorted_vals)
        if total == 0: return 0
        
        # G = (2 * sum(i * xi) / (n * sum(xi))) - (n + 1) / n
        # where i is 1-based index
        weighted_sum = sum((i + 1) * val for i, val in enumerate(sorted_vals))
        return (2 * weighted_sum) / (n * total) - (n + 1) / n
