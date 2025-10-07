# constraint.py

import numpy as np
import networkx as nx

class Constraint:
    """Base class for constraints."""
    def enforce(self, solution, graph=None):
        raise NotImplementedError("Subclasses must implement enforce().")


class TSPConstraint(Constraint):
    """
    Constraint for TSP:
    - Ensure each node is visited exactly once
    - Ensure tour starts and ends at the same node
    - If infeasible, fallback to Christofides' algorithm
    - Can also force Christofides directly with flag
    """
    def __init__(self, force_christofides=False):
        self.force_christofides = force_christofides

    def enforce(self, solution, graph=None):
        if graph is None:
            raise ValueError("Graph required for TSP constraint.")

        if self.force_christofides:
            # Always use Christofides
            return nx.approximation.traveling_salesman_problem(
                graph, cycle=True, method=nx.approximation.christofides
            )

        n = len(graph.nodes)
        visited = set()
        repaired = []

        # Step 1: try to repair solution by removing duplicates and adding missing
        for node in solution:
            if node not in visited and node in graph.nodes:
                repaired.append(node)
                visited.add(node)

        # Add missing nodes
        for node in graph.nodes:
            if node not in visited:
                repaired.append(node)

        # Close the tour
        if repaired[0] != repaired[-1]:
            repaired.append(repaired[0])

        # Step 2: check if repaired solution is a valid Hamiltonian cycle
        if len(set(repaired[:-1])) == n:
            return repaired
        else:
            # Fallback: run Christofides algorithm
            return nx.approximation.traveling_salesman_problem(
                graph, cycle=True, method=nx.approximation.christofides
            )


class MISConstraint(Constraint):
    """
    Constraint for Maximum Independent Set (MIS):
    - Ensure no two adjacent nodes are both in the solution
    """
    def enforce(self, solution, graph):
        valid_set = []
        selected = set()

        for node in solution:
            if node in graph and all(neigh not in selected for neigh in graph[node]):
                valid_set.append(node)
                selected.add(node)

        return valid_set


class ConstraintManager:
    """Handles applying constraints to decoded solutions."""
    def __init__(self, task="tsp", force_christofides=False):
        if task == "tsp":
            self.constraint = TSPConstraint(force_christofides=force_christofides)
        elif task == "mis":
            self.constraint = MISConstraint()
        else:
            raise ValueError(f"No constraints defined for task {task}")

    def apply(self, solution, graph=None):
        return self.constraint.enforce(solution, graph)


if __name__ == "__main__":
    # Example usage
    import itertools

    # Example weighted complete graph for TSP
    G = nx.complete_graph(5)
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.randint(1, 10)

    tsp_solution = [0, 2, 2, 3]  # bad solution (duplicates/missing)

    print("\n--- With Repair + Fallback ---")
    manager = ConstraintManager(task="tsp", force_christofides=False)
    repaired = manager.apply(tsp_solution, G)
    print("Original TSP:", tsp_solution)
    print("Repaired/Christofides TSP:", repaired)

    print("\n--- Force Christofides ---")
    manager_force = ConstraintManager(task="tsp", force_christofides=True)
    christofides = manager_force.apply(tsp_solution, G)
    print("Original TSP:", tsp_solution)
    print("Christofides-only TSP:", christofides)

    # Example MIS
    graph = {0: [1], 1: [0, 2], 2: [1]}
    mis_solution = [0, 1, 2]
    manager = ConstraintManager(task="mis")
    valid_set = manager.apply(mis_solution, graph)
    print("\nOriginal MIS:", mis_solution)
    print("Valid MIS:", valid_set)
