import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # ✅ stable API

def save_tsp_gif(coords, snapshots, filename="tsp_optimization.gif", duration=0.8):
    """
    Create a GIF showing TSP optimization progress with city labels.

    Args:
        coords (ndarray): Node coordinates, shape (n_nodes, 2).
        snapshots (list of lists): Sequence of tours (list of node indices).
        filename (str): Output filename for GIF.
        duration (float): Frame duration in seconds.
    """
    image_files = []

    for i, tour in enumerate(snapshots):
        plt.figure(figsize=(6,6))
        
        # Plot edges
        for j in range(len(tour)):
            n1, n2 = tour[j], tour[(j + 1) % len(tour)]
            plt.plot([coords[n1,0], coords[n2,0]],
                     [coords[n1,1], coords[n2,1]],
                     'r-', linewidth=1.5)

        # Plot nodes
        plt.scatter(coords[:,0], coords[:,1], c='blue', s=30, zorder=3)

        # Add labels (city indices)
        for idx, (x, y) in enumerate(coords):
            plt.text(x, y, str(idx), fontsize=9, ha='right', va='bottom', color="black")

        plt.title(f"TSP Optimization - Step {i+1}")
        plt.axis("equal")
        plt.axis("off")

        fname = f"frame_{i}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        image_files.append(fname)

    # Combine into GIF
    with imageio.get_writer(filename, mode="I", duration=duration) as writer:
        for fname in image_files:
            image = imageio.imread(fname)
            writer.append_data(image)

    print(f"GIF saved as {filename}")


# =============================
# Example usage
# =============================
if __name__ == "__main__":
    np.random.seed(42)
    num_nodes = 10
    coords = np.random.rand(num_nodes, 2) * 100

    # Fake snapshots (random shuffles -> final sorted tour for demo)
    snapshots = [list(np.random.permutation(num_nodes)) for _ in range(5)]
    snapshots.append(list(range(num_nodes)))  # final "clean" solution

    save_tsp_gif(coords, snapshots, filename="tsp_optimization.gif", duration=0.8)
