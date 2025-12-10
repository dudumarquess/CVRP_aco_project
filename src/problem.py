from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CVRPProblem:
    """
    Represents one instance of CVRP.

    Convention:
      - nodes are indexed from 0 to n_customers (0 is depot)
      - demands[0] = 0
      - coords.shape = (n_customers + 1, 2)
      - distance_matrix.shape = (n_customers + 1, n_customers + 1)
    """

    n_customers: int
    capacity: int
    demands: np.ndarray          # shape (n_customers + 1,)
    coords: np.ndarray           # shape (n_customers + 1, 2)
    distance_matrix: np.ndarray  # float distances

    @classmethod
    def from_data(
        cls,
        capacity: int,
        demands: np.ndarray,
        coords: np.ndarray,
    ) -> "CVRPProblem":
        """Creates the problem from ready data (good for testing)."""
        if demands.shape[0] != coords.shape[0]:
            raise ValueError("demands and coords must have the same length.")

        n_nodes = demands.shape[0]
        if n_nodes < 2:
            raise ValueError("There must be at least one customer and one depot.")

        distance_matrix = cls._euclidean_distance_matrix(coords)
        return cls(
            n_customers=n_nodes - 1,
            capacity=int(capacity),
            demands=demands.astype(int),
            coords=coords.astype(int),
            distance_matrix=distance_matrix,
        )

    @staticmethod
    def _euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
        """Computes the Euclidean distance matrix for given coordinates."""
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        return dist_matrix

    @classmethod
    def from_vrplib(cls, path: str | Path) -> "CVRPProblem":
        """Loads a CVRP problem from a VRPLIB formatted file.

        Expected sections:
          DIMENSION, CAPACITY,
          NODE_COORD_SECTION,
          DEMAND_SECTION,
          DEPOT_SECTION,
          EOF
        """

        path = Path(path)
        text = path.read_text().strip().splitlines()

        dimension = None
        capacity = None

        section = None
        coord_lines = []   # (node_idx, x, y)
        demand_lines = []  # (node_idx, demand)
        depot_index = None

        for raw_line in text:
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()

            # headers
            if upper.startswith("DIMENSION"):
                # Handles both "DIMENSION : 32" and "DIMENSION 32"
                parts = line.replace(":", " ").split()
                dimension = int(parts[-1])
                continue

            if upper.startswith("CAPACITY"):
                parts = line.replace(":", " ").split()
                capacity = int(parts[-1])
                continue

            # section switches
            if upper.startswith("NODE_COORD_SECTION"):
                section = "COORD"
                continue
            if upper.startswith("DEMAND_SECTION"):
                section = "DEMAND"
                continue
            if upper.startswith("DEPOT_SECTION"):
                section = "DEPOT"
                continue
            if upper.startswith("EOF"):
                break

            # section contents
            if section == "COORD":
                parts = line.split()
                if len(parts) < 3:
                    continue
                node_idx = int(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                coord_lines.append((node_idx, x, y))

            elif section == "DEMAND":
                parts = line.split()
                if len(parts) < 2:
                    continue
                node_idx = int(parts[0])
                d = int(parts[1])
                demand_lines.append((node_idx, d))

            elif section == "DEPOT":
                # Typical:
                # DEPOT_SECTION
                # 1
                # -1
                try:
                    idx = int(line)
                except ValueError:
                    continue
                if idx == -1:
                    continue
                depot_index = idx

        if dimension is None or capacity is None:
            raise ValueError("File missing DIMENSION or CAPACITY information.")

        coord_lines.sort(key=lambda t: t[0])
        demand_lines.sort(key=lambda t: t[0])

        if len(coord_lines) != dimension or len(demand_lines) != dimension:
            raise ValueError("Mismatch in dimension and data lines.")

        # VRPLIB uses 1..N, we map to 0..N-1
        coords_arr = np.zeros((dimension, 2), dtype=int)
        demands_arr = np.zeros(dimension, dtype=int)

        for node_idx, x, y in coord_lines:
            coords_arr[node_idx - 1] = [x, y]

        for node_idx, d in demand_lines:
            demands_arr[node_idx - 1] = d

        if depot_index is None:
            raise ValueError("DEPOT_SECTION missing depot information.")

        depot_internal = depot_index - 1
        if depot_internal != 0:
            # swap depot to index 0
            coords_arr[[0, depot_internal]] = coords_arr[[depot_internal, 0]]
            demands_arr[[0, depot_internal]] = demands_arr[[depot_internal, 0]]

        return cls.from_data(
            capacity=capacity,
            demands=demands_arr,
            coords=coords_arr,
        )

    def n_nodes(self) -> int:
        """Returns the total number of nodes (customers + depot)."""
        return self.n_customers + 1
