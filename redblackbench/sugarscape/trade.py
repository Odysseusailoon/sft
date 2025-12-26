import numpy as np
from typing import Tuple, Optional, List
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.environment import SugarEnvironment

class TradeSystem:
    """Handles trading logic between agents using MRS bargaining."""
    
    def __init__(self, env: SugarEnvironment):
        self.env = env
        
    def execute_trade_round(self, agents: List[SugarAgent]):
        """Execute a round of trading for all agents."""
        # Randomize order
        import random
        random.shuffle(agents)
        
        for agent in agents:
            if not agent.alive: continue
            
            # Find a partner
            partner = self._find_trade_partner(agent)
            if partner:
                self._bargain(agent, partner)
                
    def _find_trade_partner(self, agent: SugarAgent) -> Optional[SugarAgent]:
        """Find a random neighbor to trade with (Von Neumann neighborhood)."""
        x, y = agent.pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.env.width
            ny = (y + dy) % self.env.height
            partner = self.env.get_agent_at((nx, ny))
            if partner and partner.alive and partner != agent:
                neighbors.append(partner)
        
        if not neighbors:
            return None
            
        import random
        return random.choice(neighbors)
        
    def _bargain(self, a: SugarAgent, b: SugarAgent):
        """Execute iterative bargaining between two agents."""
        # Limit iterations to prevent infinite loops
        max_trades = 10 
        
        for _ in range(max_trades):
            mrs_a = a.mrs
            mrs_b = b.mrs
            
            # If MRS are equal (or close enough), no trade
            if abs(mrs_a - mrs_b) < 0.01:
                break
                
            # Determine direction
            # High MRS means "I value Sugar highly relative to Spice" -> Buy Sugar / Sell Spice
            # Low MRS means "I value Spice highly relative to Sugar" -> Buy Spice / Sell Sugar
            
            price = np.sqrt(mrs_a * mrs_b) # Geometric mean price (Spice per Sugar)
            
            if mrs_a > mrs_b:
                # A wants Sugar, B has Sugar (relatively). 
                # A buys Sugar from B using Spice.
                buyer, seller = a, b
            else:
                # B wants Sugar, A has Sugar.
                # B buys Sugar from A using Spice.
                buyer, seller = b, a
                
            # Calculate transaction amount
            # If p > 1: 1 unit of Sugar for p units of Spice
            # If p < 1: 1/p units of Sugar for 1 unit of Spice
            # To simplify (and follow standard impl): We trade 1 unit of the "more valuable" good?
            # Or standard Epstein Axtell: trade 1 unit of Sugar for p units of Spice if p >= 1
            # trade 1/p units of Sugar for 1 unit of Spice if p < 1
            
            sugar_amt = 0.0
            spice_amt = 0.0
            
            if price >= 1.0:
                sugar_amt = 1.0
                spice_amt = price
            else:
                sugar_amt = 1.0 / price
                spice_amt = 1.0
                
            # Check affordability
            if buyer.spice < spice_amt or seller.wealth < sugar_amt:
                break
                
            # Check Welfare Improvement (Pareto condition)
            # Calculate projected welfare
            w_buyer_current = buyer.welfare
            w_seller_current = seller.welfare
            
            # Simulated state
            # Buyer: +Sugar, -Spice
            # Seller: -Sugar, +Spice
            
            # We need to compute welfare manually without modifying state yet
            def calc_welfare(agent, s, p):
                m_t = agent.metabolism + agent.metabolism_spice
                return (s ** (agent.metabolism/m_t)) * (p ** (agent.metabolism_spice/m_t))
                
            w_buyer_new = calc_welfare(buyer, buyer.wealth + sugar_amt, buyer.spice - spice_amt)
            w_seller_new = calc_welfare(seller, seller.wealth - sugar_amt, seller.spice + spice_amt)
            
            if w_buyer_new > w_buyer_current and w_seller_new > w_seller_current:
                # Execute Trade
                buyer.wealth += int(sugar_amt) # Integer truncation for simplicity in this discrete model? 
                # Actually, standard model might use floats for resources, but our agent uses int.
                # Let's support float resources or round.
                # Since our simulation uses ints, let's stick to int exchanges if possible, or cast.
                # If we cast to int, small trades might be 0.
                # FIX: Let's assume resources are continuous (float) or convert Agent to use floats.
                # For now, let's round amounts.
                
                s_exchange = max(1, int(sugar_amt))
                p_exchange = max(1, int(spice_amt))
                
                # Re-check affordability with integers
                if buyer.spice < p_exchange or seller.wealth < s_exchange:
                    break
                    
                buyer.wealth += s_exchange
                buyer.spice -= p_exchange
                seller.wealth -= s_exchange
                seller.spice += p_exchange
                
                # Recalculate MRS for next iteration
                continue
            else:
                break
