import numpy as np
from typing import Optional, Tuple, List, Dict
from redblackbench.sugarscape.config import SugarscapeConfig
from redblackbench.sugarscape.agent import SugarAgent

class SugarEnvironment:
    """The Sugarscape environment."""
    
    def __init__(self, config: SugarscapeConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        
        # Grid state
        # capacity: max sugar at each cell
        self.sugar_capacity = np.zeros((self.width, self.height), dtype=int)
        # amount: current sugar at each cell
        self.sugar_amount = np.zeros((self.width, self.height), dtype=int)
        
        # Spice layers (initialized only if enabled)
        self.spice_capacity = np.zeros((self.width, self.height), dtype=int)
        self.spice_amount = np.zeros((self.width, self.height), dtype=int)
        
        # Agent tracking
        # Map from (x, y) -> SugarAgent
        self.grid_agents: Dict[Tuple[int, int], SugarAgent] = {}
        
        self._init_landscape()
        
    def _init_landscape(self):
        """Initialize sugar and spice capacity with dual peaks."""
        # Sugar Peak centers (Classic: NE and SW)
        sugar_p1 = (15, 15)
        sugar_p2 = (35, 35)
        
        # Spice Peak centers (Opposite: NW and SE)
        spice_p1 = (15, 35)
        spice_p2 = (35, 15)
        
        for x in range(self.width):
            for y in range(self.height):
                # 1. Sugar Generation
                d1 = self._dist((x, y), sugar_p1)
                d2 = self._dist((x, y), sugar_p2)
                min_dist_s = min(d1, d2)
                
                cap_s = self._dist_to_capacity(min_dist_s)
                cap_s = min(cap_s, self.config.max_sugar_capacity)
                
                self.sugar_capacity[x, y] = cap_s
                self.sugar_amount[x, y] = cap_s
                
                # 2. Spice Generation (if enabled)
                if self.config.enable_spice:
                    d3 = self._dist((x, y), spice_p1)
                    d4 = self._dist((x, y), spice_p2)
                    min_dist_p = min(d3, d4)
                    
                    cap_p = self._dist_to_capacity(min_dist_p)
                    cap_p = min(cap_p, self.config.max_spice_capacity)
                    
                    self.spice_capacity[x, y] = cap_p
                    self.spice_amount[x, y] = cap_p

    def _dist_to_capacity(self, dist: float) -> int:
        """Map distance to resource capacity (Classic steps)."""
        if dist <= 5: return 4
        elif dist <= 10: return 3
        elif dist <= 15: return 2
        elif dist <= 20: return 1
        else: return 0
                
    def _dist(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Euclidean distance on torus."""
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        
        dx = min(dx, self.width - dx)
        dy = min(dy, self.height - dy)
        
        return np.sqrt(dx**2 + dy**2)

    def growback(self):
        """Regenerate sugar and spice."""
        # 1. Sugar
        alpha_s = self.config.sugar_growback_rate
        self.sugar_amount = np.minimum(
            self.sugar_amount + alpha_s,
            self.sugar_capacity
        )
        
        # 2. Spice (if enabled)
        if self.config.enable_spice:
            alpha_p = self.config.spice_growback_rate
            self.spice_amount = np.minimum(
                self.spice_amount + alpha_p,
                self.spice_capacity
            )

    def get_agent_at(self, pos: Tuple[int, int]) -> Optional[SugarAgent]:
        return self.grid_agents.get(pos)

    def get_sugar_at(self, pos: Tuple[int, int]) -> int:
        return self.sugar_amount[pos]
        
    def get_spice_at(self, pos: Tuple[int, int]) -> int:
        if not self.config.enable_spice: return 0
        return self.spice_amount[pos]

    def harvest_sugar(self, pos: Tuple[int, int]) -> int:
        amount = self.sugar_amount[pos]
        self.sugar_amount[pos] = 0
        return amount
        
    def harvest_spice(self, pos: Tuple[int, int]) -> int:
        if not self.config.enable_spice: return 0
        amount = self.spice_amount[pos]
        self.spice_amount[pos] = 0
        return amount

    def get_local_density(self, pos: Tuple[int, int], radius: int = 1) -> float:
        """Calculate local agent density in Moore neighborhood."""
        count = 0
        total_spots = 0
        x, y = pos
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0: continue
                
                nx = (x + dx) % self.width
                ny = (y + dy) % self.height
                total_spots += 1
                
                if (nx, ny) in self.grid_agents:
                    count += 1
                    
        return count / total_spots if total_spots > 0 else 0.0

    def get_site_quality(self, pos: Tuple[int, int]) -> float:
        """Get long-term site quality (capacity)."""
        quality = self.sugar_capacity[pos]
        if self.config.enable_spice:
            # Simple sum or max? Sum represents total resource potential
            quality += self.spice_capacity[pos]
        return float(quality)

    def add_agent(self, agent: SugarAgent):
        if agent.pos in self.grid_agents:
            raise ValueError(f"Position {agent.pos} already occupied")
        self.grid_agents[agent.pos] = agent

    def remove_agent(self, agent: SugarAgent):
        if self.grid_agents.get(agent.pos) == agent:
            del self.grid_agents[agent.pos]

    def move_agent(self, agent: SugarAgent, new_pos: Tuple[int, int]):
        if new_pos in self.grid_agents:
            raise ValueError(f"Position {new_pos} already occupied")
        
        del self.grid_agents[agent.pos]
        self.grid_agents[new_pos] = agent
        agent.pos = new_pos

    def is_occupied(self, pos: Tuple[int, int]) -> bool:
        return pos in self.grid_agents
        
    def get_random_unoccupied_pos(self, rng) -> Tuple[int, int]:
        """Find a random unoccupied position."""
        # Simple rejection sampling
        # Might be slow if grid is very full, but N=250 on 50x50 (2500 cells) is 10% density.
        while True:
            x = rng.randint(0, self.width - 1)
            y = rng.randint(0, self.height - 1)
            if not self.is_occupied((x, y)):
                return (x, y)
