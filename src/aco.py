from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .problem import CVRPProblem
from .solution import Route, Solution

@dataclass
class ACSParams:
    """
    Parameters for the Ant Colony System (ACS) for CVRP.
    """
    n_ants: int = 10
    n_iterations: int = 100
    alpha: float = 1.0          # pheromone importance
    beta: float = 2.0           # heuristic importance
    rho: float = 0.1            # pheromone evaporation rate
    xi: float = 0.1             # local pheromone decay coefficient
    q0: float = 0.9             # probability of exploitation vs exploration
    tau0: float = 0.01          # initial pheromone level
    
class AntColony:
    def __init__(self, problem: CVRPProblem, params: ACSParams):
        self.problem = problem
        self.params = params
        
        n = problem.n_nodes()
        
        # initialize pheromone matrix
        self.tau = np.full((n, n), params.tau0, dtype=float)
        
        # initialize heuristic matrix (1/distance)
        self.eta = np.zeros((n, n), dtype=float)
        with np.errstate(divide='ignore'):
            self.eta = 1.0 / problem.distance_matrix
            self.eta[np.isinf(self.eta)] = 0.0  # handle division by zero
            
        self.rng = np.random.default_rng(seed=42)  # for reproducibility
        self.global_best: Optional[Solution] = None
        
    # Principal API
    
    def run(self) -> Solution:
        """
        Executes the ACS algorithm and returns the best found solution.
        """
        for iteration in range(self.params.n_iterations):
            solutions: List[Solution] = []
            
            for _ in range(self.params.n_ants):
                sol = self._construct_solution()
                
                # we are going to change to local search after, for now just store
                solutions.append(sol)
                
            best_iteration = min(solutions, key=lambda s: s.total_cost(self.problem))
            
            if (self.global_best is None or best_iteration.total_cost(self.problem) < 
                self.global_best.total_cost(self.problem)):
                self.global_best = best_iteration
                
            self._global_pheromone_update(self.global_best)
            
            print(
                f"- Iteration {iteration+1}/{self.params.n_iterations}, "
                f"- Best iteration: {best_iteration.total_cost(self.problem):.2f}"
                f"- Global best: {self.global_best.total_cost(self.problem):.2f}"
            )
            
            return self.global_best
        
        # Construction of the solution with capacity constraints
    def _construct_solution(self) -> Solution:
        """
        Constructs a solution using the ACS probabilistic rules with capacity contraints.
        """
        n_nodes = self.problem.n_nodes()
        customers = set(range(1, self.problem.n_customers + 1))
        
        routes: List[Route] = []
        
        while customers:
            route_nodes = [0]  # start at depot
            current_node = 0
            current_load = 0
            
            while True:
                # clients that can be visited
                candidates = []
                for j in customers:
                    demand_j = int(self.problem.demands[j])
                    if current_load + demand_j <= self.problem.capacity:
                        candidates.append(j)
                    
                if not candidates:
                    route_nodes.append(0)  # return to depot
                    routes.append(Route(nodes=route_nodes))
                    break
                
                next_node = self._choose_next_node(current_node, candidates)
                
                #updates local pheromone
                self._local_pheromone_update(current_node, next_node)
                
                route_nodes.append(next_node)
                current_load += int(self.problem.demands[next_node])
                customers.remove(next_node)
                current_node = next_node
                
            # end of the route; if there are still customers, start a new route
            
        sol = Solution(routes=routes)
        # sanity check
        if not sol.is_feasible(self.problem):
            raise RuntimeError("Constructed solution is not feasible!")
        return sol
    
    # ACS rules
    def _choose_next_node(self, current: int, candidates: List[int]) -> int:
        """
        with probability q0 choose the best next node (argmax - exploitation), if not choose probabilistically (exploration)
        """
        tau = self.tau
        eta = self.eta
        alpha = self.params.alpha
        beta = self.params.beta
        
        values = []
        for j in candidates:
            val = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
            values.append(val)
        values = np.array(values, dtype=float)
        
        if values.sum() == 0:
            return self.rng.choice(candidates)
        
        q = self.rng.random()
        
        if q <= self.params.q0:
            # exploitation
            idx = int(np.argmax(values))
            return candidates[idx]
        else:
            # exploration
            probs = values / values.sum()
            idx = self.rng.choice(len(candidates), p=probs)
            return candidates[idx]
        
    def _local_pheromone_update(self, i: int, j: int) -> None:
        """
        local pheromone update after an ant moves from i to j
        tau(i,j) = (1 - xi) * tau(i,j) + xi * tau0
        """
        xi = self.params.xi
        tau0 = self.params.tau0
        
        self.tau[i, j] = (1.0 - xi) * self.tau[i, j] + xi * tau0
        self.tau[j, i] = self.tau[i, j]  # symmetric graph
        
    def _global_pheromone_update(self, best_solution: Solution) -> None:
        """
        global pheromone update using the best solution found
        """
        rho = self.params.rho
        
        # evaporation
        self.tau *= (1.0 - rho)
        
        # reiforcement
        L_best = best_solution.total_cost(self.problem)
        if L_best <= 0:
            return
        
        delta = rho * (1.0 / L_best)
        
        for route in best_solution.routes:
            for i in range(len(route.nodes) - 1):
                a = route.nodes[i]
                b = route.nodes[i + 1]
                self.tau[a, b] += delta
                self.tau[b, a] = self.tau[a, b]  # symmetric graph
    
                