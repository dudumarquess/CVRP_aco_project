from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Callable, Dict, Any

import numpy as np

from .problem import CVRPProblem
from .solution import Route, Solution

Observer = Callable[[Dict[str, Any]], None]

@dataclass
class ACSParams:
    """
    Parameters for the Ant Colony System (ACS) for CVRP.
    """
    n_ants: int = 10
    n_iterations: int = 30
    alpha: float = 1.0          # pheromone importance
    beta: float = 2.3           # heuristic importance
    rho: float = 0.1            # pheromone evaporation rate
    q0: float = 0.9             # probability of exploitation vs exploration
    tau0: float = 0.01          # initial pheromone level
    candidate_list_mode: str = "fixed"  # "none", "fixed", "sqrt"
    candidate_list_size: int = 15  # k-nearest neighbors considered when selecting next node
    use_q0_schedule: bool = False
    q0_min: float = 0.7
    q0_max: float = 0.95
    
class AntColony:
    def __init__(
        self,
        problem: CVRPProblem,
        params: ACSParams,
        observer: Optional[Observer] = None,
        seed: int = 42,
        local_search_fn: Optional[Callable[[Solution, CVRPProblem], Solution]] = None,
        local_search_mode: str = "none",  # "none", "best"
    ):
        self.problem = problem
        self.params = params
        self.observer = observer
        self.best_history: List[float] = []
        self.pheromone_history: List[Dict[str, float]] = []
        self.candidate_list: List[List[int]] = []
        self.candidate_list_k_effective = 0
        self.local_search_fn = local_search_fn
        self.local_search_mode = local_search_mode
        self.ls_stats: Dict[str, float | int] = {}
        self.q0_current = params.q0
        
        n = problem.n_nodes()
        
        # initialize pheromone matrix
        self.tau = np.full((n, n), params.tau0, dtype=float)
        
        # initialize heuristic matrix (1/distance)
        self.eta = np.zeros((n, n), dtype=float)
        with np.errstate(divide="ignore"):
            self.eta = 1.0 / problem.distance_matrix
            self.eta[np.isinf(self.eta)] = 0.0  # handle division by zero

        self.rng = np.random.default_rng(seed=seed)  # for reproducibility
        self.global_best: Optional[Solution] = None

        # precompute per-node candidate lists (k-nearest neighbors)
        mode = (self.params.candidate_list_mode or "fixed").lower()
        if mode == "none":
            k = 0
        elif mode == "fixed":
            k = max(0, int(self.params.candidate_list_size))
        elif mode == "sqrt":
            k = int(math.ceil(math.sqrt(max(1, self.problem.n_customers))))
            k = max(1, k)
        else:
            raise ValueError(f"Unknown candidate_list_mode: {self.params.candidate_list_mode}")

        self.candidate_list_k_effective = k
        if k > 0:
            self.candidate_list = []
            for i in range(n):
                dists = self.problem.distance_matrix[i]
                order = np.argsort(dists)
                neighbors = [int(idx) for idx in order if idx != i and idx != 0]
                self.candidate_list.append(neighbors[:k])
        else:
            self.candidate_list = [[] for _ in range(n)]
        
    # Principal API
    
    def run(self) -> Solution:
        self.best_history = []
        self.pheromone_history = []
        total_ls_applied = 0

        for iteration in range(self.params.n_iterations):
            # adapt q0 at iteration start
            if self.params.use_q0_schedule:
                frac = 0.0 if self.params.n_iterations <= 1 else iteration / (self.params.n_iterations - 1)
                self.q0_current = self.params.q0_min + (self.params.q0_max - self.params.q0_min) * frac
            else:
                self.q0_current = self.params.q0

            solutions: List[Solution] = []
            solutions_costs: List[float] = []

            for _ in range(self.params.n_ants):
                sol = self._construct_solution()
                solutions.append(sol)
                solutions_costs.append(sol.total_cost(self.problem))

            # Local search (optional) on the best solution of the iteration
            if self.local_search_fn is not None and self.local_search_mode == "best":
                best_idx = int(np.argmin(solutions_costs))
                improved = self.local_search_fn(solutions[best_idx], self.problem)
                improved_cost = improved.total_cost(self.problem)
                solutions[best_idx] = improved
                solutions_costs[best_idx] = improved_cost
                total_ls_applied += 1

            # Re-evaluate best after possible local search
            best_idx = int(np.argmin(solutions_costs))
            best_iteration = solutions[best_idx]
            best_iteration_cost = solutions_costs[best_idx]

            if self.global_best is None or best_iteration_cost < self.global_best.total_cost(self.problem):
                self.global_best = best_iteration

            self._global_pheromone_update(self.global_best)

            best_it_cost = best_iteration_cost
            best_glb_cost = self.global_best.total_cost(self.problem)
            self.best_history.append(best_glb_cost)

            tau_mean = float(self.tau.mean())
            tau_max = float(self.tau.max())
            concentration = float(tau_max / (tau_mean + 1e-12))
            self.pheromone_history.append({
                "iteration": iteration + 1,
                "tau_mean": tau_mean,
                "tau_max": tau_max,
                "concentration": concentration,
            })

            if self.observer is not None:
                self.observer({
                    "type": "iteration_end",
                    "iteration": iteration + 1,
                    "best_iteration_cost": best_it_cost,
                    "global_best_cost": best_glb_cost,
                    "global_best_solution": self.global_best,  # snapshot
                    "tau_mean": tau_mean,
                    "tau_max": tau_max,
                    "concentration": concentration,
                })

            print(
                f"- Iteration {iteration+1}/{self.params.n_iterations}, "
                f"- Best iteration: {best_it_cost:.2f} "
                f"- Global best: {best_glb_cost:.2f}"
            )

        self.ls_stats = {
            "total_ls_applications": total_ls_applied,
            "local_search_mode": self.local_search_mode,
            "ls_per_iter": total_ls_applied / max(1, self.params.n_iterations),
        }

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
        # classic candidate list: precomputed k-nearest neighbors per node
        candset = set(candidates)
        cl = [j for j in self.candidate_list[current] if j in candset]

        selected = cl if cl else candidates

        tau = self.tau
        eta = self.eta
        alpha = self.params.alpha
        beta = self.params.beta

        values = []
        for j in selected:
            val = (tau[current, j] ** alpha) * (eta[current, j] ** beta)
            values.append(val)
        values = np.array(values, dtype=float)
        
        if not selected:
            raise RuntimeError("No candidates to choose from!")
        
        if values.sum() == 0:
            return self.rng.choice(selected)
        
        q = self.rng.random()
        
        if q <= self.q0_current:
            # exploitation
            idx = int(np.argmax(values))
            return selected[idx]
        else:
            # exploration
            probs = values / values.sum()
            idx = self.rng.choice(len(selected), p=probs)
            return selected[idx]
        
    def _local_pheromone_update(self, i: int, j: int) -> None:
        """
        local pheromone update after an ant moves from i to j
        tau(i,j) = (1 - rho) * tau(i,j) + rho * tau0
        """
        rho = self.params.rho
        tau0 = self.params.tau0
        self.tau[i, j] = (1.0 - rho) * self.tau[i, j] + rho * tau0
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
    
                
