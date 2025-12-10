from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from .problem import CVRPProblem

@dataclass
class Route:
    """
    represents a single vehicle route in a CVRP solution.
    starts and ends at the depot (node 0).
    
    exemple: Route([0, 3, 5, 2, 0]) 
    """
    nodes: List[int] = field(default_factory=list)
    
    def cost(self, problem: CVRPProblem) -> float:
        """Calculates the total cost (distance) of this route based on the problem's distance matrix."""
        dist = 0.0
        for i in range(len(self.nodes) - 1):
            a = self.nodes[i]
            b = self.nodes[i + 1]
            dist += problem.distance_matrix[a, b]
        return dist
    
    def load(self, problem: CVRPProblem) -> int:
        """sum all demands of customers in this route. (ignore depot)"""
        return int(sum(problem.demands[node] for node in self.nodes if node != 0))
    
    def customers(self) -> Set[int]:
        """returns the set of customer nodes in this route (excluding depot)."""
        return {n for n in self.nodes if n != 0}
    
@dataclass
class Solution:
    """
    Set of routes (vehicles). each route must start and finish at node 0 (depot)
    """
    routes: List[Route] = field(default_factory=list)
    
    def total_cost(self, problem: CVRPProblem) -> float:
        """Calculates the total cost (distance) of all routes in the solution."""
        return sum(route.cost(problem) for route in self.routes)
    
    def all_customers(self) -> Set[int]:
        """Returns the set of all customer nodes served in this solution."""
        customers = set()
        for route in self.routes:
            customers.update(route.customers())
        return customers
    
    def is_feasible(self, problem: CVRPProblem) -> bool:
        """Checks if the solution is feasible:
        - all customers are served exactly once
        - no route exceeds vehicle capacity
        """
        #capacity
        for route in self.routes:
            if route.load(problem) > problem.capacity:
                return False
            
        visited = []
        for route in self.routes:
            visited.extend(n for n in route.nodes if n != 0)
            
        visited_set = set(visited)
        expected_set = set(range(1, problem.n_customers + 1))
        
        if visited_set != expected_set:
            return False
        
        if len(visited) != len(expected_set):
            return False
        
        if len(visited) != len(visited_set):
            return False    
        
        return True
       
        