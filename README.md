# Constraint-Aware DIFUSCO: Graph-Based Diffusion Models for Combinatorial Optimization

This repository contains the implementation of **Constraint-Aware DIFUSCO**, an extension of the original [DIFUSCO](https://github.com/sekulicd/difusco) framework for solving **combinatorial optimization problems** such as the **Travelling Salesman Problem (TSP)**.  
The project introduces **constraint-aware decoding** and **Christofides-guided integration** to improve **solution feasibility** and **constraint satisfaction** in graph-based diffusion solvers.

---

## Overview

The **Constraint-Aware DIFUSCO** framework combines:
- **Graph diffusion models** (based on PyTorch Geometric) to denoise graph structures into feasible solutions.
- **Constraint-aware decoding**, a decoding module that enforces problem-specific constraints (e.g., visiting all nodes exactly once).
- **Christofides algorithm integration** to bias the diffusion process toward feasible tours or repair invalid ones post-generation.

---

## Key Features

- **Constraint-Aware Decoding**  
  Enhances solution feasibility by pruning infeasible edges during the diffusion process.  

- **Christofides Integration**  
  Implemented in two modes:
  1. **Post-processing repair** — applies Christofides on DIFUSCO outputs to ensure tour validity.
  2. **Guided decoding** — integrates Christofides during generation to bias sampling toward feasible paths.

- **Data Support**  
  Trained and evaluated on:
  - [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)

- **Scalable Training**  
  Supports large datasets and long training runs (20–50 epochs).  
  Requires **high computational resources** due to iterative denoising and edge constraint evaluation.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/constraint-aware-difusco.git
cd constraint-aware-difusco
pip install -r requirements.txt
