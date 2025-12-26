from dataclasses import dataclass
import random
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from redblackbench.sugarscape.environment import SugarEnvironment

@dataclass
class SugarAgent:
    """An agent in the Sugarscape."""
    
    agent_id: int
    pos: Tuple[int, int]
    
    # Fixed attributes
    vision: int
    metabolism: int
    max_age: int
    
    # Variable state
    wealth: int
    spice: int = 0
    metabolism_spice: int = 0
    
    age: int = 0
    persona: str = "A" # Default to Conservative
    
    def __post_init__(self):
        self.alive = True
        # Metrics tracking
        self.initial_pos = self.pos
        self.visited_cells = {self.pos}
        self.metrics = {
            "displacement": 0.0,
            "unique_visited": 1
        }
    
    @property
    def welfare(self) -> float:
        """Calculate Cobb-Douglas Welfare/Utility."""
        if self.metabolism_spice == 0:
            return float(self.wealth)
        
        # W = w_s^(m_s/m_total) * w_p^(m_p/m_total)
        # To avoid overflow/underflow, we can use log form or just direct calculation if numbers are small
        m_total = self.metabolism + self.metabolism_spice
        if m_total == 0: return 0.0
        
        return (self.wealth ** (self.metabolism / m_total)) * (self.spice ** (self.metabolism_spice / m_total))

    @property
    def mrs(self) -> float:
        """Calculate Marginal Rate of Substitution (Spice for Sugar)."""
        # MRS = (MU_sugar / MU_spice) = (m_s * w_p) / (m_p * w_s)
        if self.metabolism_spice == 0 or self.wealth == 0:
            return 9999.0 # Infinite demand for sugar or no spice value
        if self.spice == 0:
            return 0.0001 # Almost zero value for sugar relative to spice (needs spice urgently)
            
        return (self.metabolism * self.spice) / (self.metabolism_spice * self.wealth)

    def step(self, env: "SugarEnvironment"):
        """Execute one simulation step for the agent."""
        if not self.alive:
            return

        # 1. Move and Harvest
        self._move_and_harvest(env)
        
        # Update movement metrics
        self._update_metrics(env)
        
        # 2. Metabolize
        self.wealth -= self.metabolism
        if env.config.enable_spice:
            self.spice -= self.metabolism_spice
        
        # 3. Age
        self.age += 1
        
        # 4. Check death
        # Die if EITHER resource is depleted
        if self.wealth <= 0 or (env.config.enable_spice and self.spice <= 0) or self.age >= self.max_age:
            self.alive = False
            # Remove from environment is handled by the simulation loop or environment
            
    def _move_and_harvest(self, env: "SugarEnvironment"):
        """Move to the best location within vision and harvest sugar."""
        # 1. Identify candidate spots
        candidates = self._get_visible_spots(env)
        
        # 2. Filter occupied spots (except self)
        # Note: In standard Sugarscape, agents can't move to occupied spots.
        # But staying in current spot is allowed (which is "occupied" by self).
        valid_candidates = []
        for pos in candidates:
            agent_at_pos = env.get_agent_at(pos)
            if agent_at_pos is None or agent_at_pos is self:
                valid_candidates.append(pos)
        
        if not valid_candidates:
            # No valid moves (shouldn't happen if current pos is included)
            return

        # 3. Choose best spot
        # Criteria: Max sugar -> Min distance -> Random
        
        # Get sugar amounts
        # Note: Agent sees current sugar level, not capacity
        candidates_with_info = []
        for pos in valid_candidates:
            # Base data
            dist = self._get_distance(self.pos, pos, env.width, env.height)
            
            if env.config.enable_personas:
                metric = self._calculate_persona_score(pos, dist, env)
            else:
                # Standard Logic (Wealth/Welfare Maximization)
                sugar = env.get_sugar_at(pos)
                spice = env.get_spice_at(pos)
                
                potential_wealth = self.wealth + sugar
                potential_spice = self.spice + spice
                
                if env.config.enable_spice:
                    m_total = self.metabolism + self.metabolism_spice
                    welfare = (potential_wealth ** (self.metabolism / m_total)) * (potential_spice ** (self.metabolism_spice / m_total))
                    metric = welfare
                else:
                    metric = potential_wealth
                
            candidates_with_info.append((pos, metric, dist))
        
        # Find max metric
        max_metric = max(c[1] for c in candidates_with_info)
        best_metric_candidates = [c for c in candidates_with_info if c[1] == max_metric]
        
        # Find min distance among max metric
        min_dist = min(c[2] for c in best_metric_candidates)
        best_candidates = [c for c in best_metric_candidates if c[2] == min_dist]
        
        # Random choice among ties
        target = random.choice(best_candidates)
        target_pos = target[0]
        
        # 4. Move
        if target_pos != self.pos:
            env.move_agent(self, target_pos)
        
        # 5. Harvest
        harvested_s = env.harvest_sugar(self.pos)
        self.wealth += harvested_s
        
        if env.config.enable_spice:
            harvested_p = env.harvest_spice(self.pos)
            self.spice += harvested_p

    def _calculate_persona_score(self, pos: Tuple[int, int], dist: int, env: "SugarEnvironment") -> float:
        """Calculate score based on Persona logic."""
        # 1. Gather Context
        sugar_at = env.get_sugar_at(pos)
        spice_at = env.get_spice_at(pos)
        
        # Calculate Utility/Wealth at target
        w_pot = self.wealth + sugar_at
        s_pot = self.spice + spice_at
        
        if env.config.enable_spice:
            m_total = self.metabolism + self.metabolism_spice
            utility_now = (w_pot ** (self.metabolism / m_total)) * (s_pot ** (self.metabolism_spice / m_total))
        else:
            utility_now = float(w_pot)
            
        local_density = env.get_local_density(pos)
        site_quality = env.get_site_quality(pos)
        
        # Hyperparameters
        beta = env.config.long_term_weight
        kappa = env.config.crowding_penalty
        lam = env.config.exploration_factor
        
        # Survival Check
        # S* = mult * metabolism
        mult = env.config.safety_threshold_mult
        safe_w = mult * self.metabolism
        safe_s = mult * self.metabolism_spice
        
        is_survival_mode = False
        if env.config.enable_spice:
            if self.wealth < safe_w or self.spice < safe_s:
                is_survival_mode = True
        else:
            if self.wealth < safe_w:
                is_survival_mode = True
        
        # Persona Logic
        if self.persona == "A": # Conservative
            if is_survival_mode:
                # Avoid long moves, prioritize safety
                return utility_now - 0.8 * dist - kappa * local_density
            else:
                return 1.0 * utility_now - 0.6 * dist - kappa * local_density + 0.2 * beta * site_quality + lam * dist
                
        elif self.persona == "B": # Foresight
            if is_survival_mode:
                return utility_now - 0.5 * dist # Fallback to greedy
            
            return (0.75 * utility_now + 
                    0.75 * beta * site_quality - 
                    0.45 * dist - 
                    kappa * local_density + 
                    0.15 * lam * dist)
                    
        elif self.persona == "C": # Nomad
            # Novelty proxy: distance (moving far is novel) + unvisited bonus
            novelty = dist
            if pos not in self.visited_cells:
                novelty += 10  # Bonus for unvisited cells

            if is_survival_mode:
                 # Check strict death condition: if move implies death next turn?
                 # Simplified: just return utility to survive
                 return utility_now
                 
            return (0.65 * utility_now + 
                    0.35 * beta * site_quality - 
                    0.25 * dist - 
                    kappa * local_density + 
                    0.55 * lam * novelty)
                    
        elif self.persona == "D": # Risk-taker
             # Hard constraint: if death guaranteed next turn, avoid?
             # For scoring, just maximize yield
             return (1.15 * utility_now - 
                     0.25 * dist - 
                     kappa * local_density + 
                     0.05 * beta * site_quality + 
                     0.10 * lam * dist)
        
        return utility_now

    def _get_visible_spots(self, env: "SugarEnvironment") -> List[Tuple[int, int]]:
        """Get all visible spots in 4 cardinal directions."""
        spots = [self.pos] # Can always stay put
        
        x, y = self.pos
        w, h = env.width, env.height
        
        for d in range(1, self.vision + 1):
            # North (y - d)
            spots.append((x, (y - d) % h))
            # South (y + d)
            spots.append((x, (y + d) % h))
            # East (x + d)
            spots.append(((x + d) % w, y))
            # West (x - d)
            spots.append(((x - d) % w, y))
            
        return spots

    def _get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int], w: int, h: int) -> int:
        """Calculate Manhattan distance on a torus."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        # Wrap-around distance
        dx = min(dx, w - dx)
        dy = min(dy, h - dy)
        
        return dx + dy

    def _update_metrics(self, env: "SugarEnvironment"):
        """Update agent metrics."""
        # Calculate displacement from origin
        self.metrics["displacement"] = self._get_distance(self.initial_pos, self.pos, env.width, env.height)
        
        # Track visited cells
        self.visited_cells.add(self.pos)
        self.metrics["unique_visited"] = len(self.visited_cells)
