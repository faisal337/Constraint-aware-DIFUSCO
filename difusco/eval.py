import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

# =============================
# Helper functions
# =============================

def is_valid_tour(tour, num_nodes):
    """Check if a tour is valid TSP solution."""
    return len(tour) == num_nodes and len(set(tour)) == num_nodes


def tour_length(tour, distance_matrix):
    length = 0
    for i in range(len(tour)):
        length += distance_matrix[tour[i], tour[(i+1) % len(tour)]]
    return length


def evaluate_solutions(all_tours, distance_matrix):
    num_nodes = distance_matrix.shape[0]
    validity_flags = [is_valid_tour(tour, num_nodes) for tour in all_tours]
    valid_tours = [tour for i, tour in enumerate(all_tours) if validity_flags[i]]
    lengths = [tour_length(tour, distance_matrix) for tour in valid_tours]
    
    metrics = {
        'total': len(all_tours),
        'feasible': sum(validity_flags),
        'infeasible': len(all_tours) - sum(validity_flags),
        'min_length': np.min(lengths) if lengths else None,
        'max_length': np.max(lengths) if lengths else None,
        'mean_length': np.mean(lengths) if lengths else None
    }
    return metrics, valid_tours, lengths

# =============================
# Plotting functions
# =============================

def plot_feasibility(metrics, solver_name):
    plt.bar(["Feasible", "Infeasible"], [metrics['feasible'], metrics['infeasible']])
    plt.title(f"Solution Feasibility - {solver_name}")
    plt.savefig("feasibility_hist.png")
    plt.show()


def plot_length_hist(lengths, solver_name):
    plt.hist(lengths, bins=20, alpha=0.7)
    plt.xlabel("Tour Length")
    plt.ylabel("Frequency")
    plt.title(f"Tour Length Distribution - {solver_name}")
    plt.savefig("plot_length_hist.png")
    plt.show()


def plot_example_tour(tour, coords, solver_name):
    G = nx.complete_graph(len(coords))
    pos = {i: tuple(coords[i]) for i in range(len(coords))}
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edgelist=[(tour[i], tour[(i+1)%len(tour)]) for i in range(len(tour))], edge_color='r', width=2)
    plt.title(f"Example TSP Tour - {solver_name}")
    plt.savefig("plot_example_tour.png")
    plt.show()

# =============================
# Comparison Table
# =============================

def print_comparison_table(metrics_dict):
    table_data = []
    for solver, metrics in metrics_dict.items():
        table_data.append([
            solver,
            metrics['feasible'],
            metrics['infeasible'],
            metrics['min_length'],
            metrics['mean_length'],
            metrics['max_length']
        ])
    headers = ["Solver", "Feasible", "Infeasible", "Min Length", "Mean Length", "Max Length"]
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

# =============================
# Example Usage
# =============================

if __name__ == '__main__':
    # Example distance matrix and coordinates
    num_nodes = 5
    coords = np.random.rand(num_nodes, 2) * 100
    distance_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(-1))

    # Example tours from different solvers (replace with actual outputs)
    difusco_tours = [list(np.random.permutation(num_nodes)) for _ in range(10)]
    ca_tours = [list(np.random.permutation(num_nodes)) for _ in range(10)]  # Constraint-aware
    concorde_tours = [list(np.random.permutation(num_nodes)) for _ in range(10)]

    metrics_dict = {}

    # Evaluate DIFUSCO
    metrics, valid_tours, lengths = evaluate_solutions(difusco_tours, distance_matrix)
    metrics_dict['DIFUSCO'] = metrics
    plot_feasibility(metrics, 'DIFUSCO')
    plot_length_hist(lengths, 'DIFUSCO')
    if valid_tours: plot_example_tour(valid_tours[0], coords, 'DIFUSCO')

    # Evaluate Constraint-Aware DIFUSCO
    metrics, valid_tours, lengths = evaluate_solutions(ca_tours, distance_matrix)
    metrics_dict['Constraint-Aware DIFUSCO'] = metrics
    plot_feasibility(metrics, 'Constraint-Aware DIFUSCO')
    plot_length_hist(lengths, 'Constraint-Aware DIFUSCO')
    if valid_tours: plot_example_tour(valid_tours[0], coords, 'Constraint-Aware DIFUSCO')

    # Evaluate Concorde
    metrics, valid_tours, lengths = evaluate_solutions(concorde_tours, distance_matrix)
    metrics_dict['Concorde'] = metrics
    plot_feasibility(metrics, 'Concorde')
    plot_length_hist(lengths, 'Concorde')
    if valid_tours: plot_example_tour(valid_tours[0], coords, 'Concorde')

    # Print Comparison Table
    print_comparison_table(metrics_dict)

import numpy as np
import os

all_tours = []
all_coords = []

# inside evaluation loop
for instance in dataset:
    coords = instance['coords']
    tour = solver.solve(coords)   # or model.generate(instance)

    all_tours.append(tour)
    all_coords.append(coords)

# after evaluation is done:
save_dir = "results/solutions"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "tours.npy"), np.array(all_tours, dtype=object))
np.save(os.path.join(save_dir, "coords.npy"), np.array(all_coords, dtype=object))


