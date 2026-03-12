# 🧪 Lab: Mathematical Foundations in Python

This repository contains a collection of algorithmic implementations focusing on core concepts from the book **Mathematics for Machine Learning (MML)**. Each project is built from scratch using Python and NumPy to solidify the bridge between abstract linear algebra/calculus and practical software engineering.

---

## 🏗️ Project Modules

### 1. Automatic Differentiation (Reverse Mode)
**File:** `automatic_differentiation.py`  
A custom engine for **Reverse Mode Automatic Differentiation**, the backbone of the Backpropagation algorithm used in Deep Learning.
* **Core Logic:** Uses a `Node` class to build a Directed Acyclic Graph (DAG). Each node caches its local derivative during the forward pass.
* **Summation Rule:** Correctly handles the multivariable chain rule by accumulating gradients (`node.grad += ...`) when a variable contributes to multiple paths (e.g., $f(x) = x^2 + 2x$).
* **Symbolic Trace:** Includes a logging system to visualize the Chain Rule propagation step-by-step through the graph.

### 2. Recursive Laplace Expansion
**File:** `laplace_expansion_determinant.py`  
An implementation of the **Laplace Expansion** (Cofactor Expansion) to compute the determinant of $n \times n$ matrices.
* **Optimization:** Includes a heuristic function `find_most_zeros_row_index` to identify the row with the most zeros, significantly reducing the number of recursive calls required.
* **Concept:** Demonstrates the recursive nature of sub-matrix minors and the alternating sign logic $(-1)^{i+j}$ used in computing determinants of higher-order matrices.

### 3. Hub and Spoke Network Analysis
**File:** `hub_and_spoke_optimizer.py`  
An application of **Spectral Theory** and **Eigen-centrality** to logistics and infrastructure optimization.
* **Use Case:** Analyzing connectivity between Pakistani cities (Islamabad, Karachi, Swat, etc.) based on inverse-distance weights.
* **Algorithm:** Computes the **Principal Eigenvector** of a symmetric adjacency matrix ($A^T A$).
* **Outcome:** Identifies the "Hub" city with the highest network reach—the most efficient transfer point for refueling and maintenance in a transportation system.

---

## 🛠️ Technical Stack
* **Python 3.10+**
* **NumPy:** Used for efficient matrix operations and eigensolvers.
* **MML Textbook:** Theoretical grounding based on Chapters 2 (Linear Algebra), 4 (Matrix Decomposition), and 5 (Vector Calculus).

---

## ⚙️ How to Run

This repository uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package and project management. Each project is a standalone script.

### Using uv (Recommended)
You can run any script directly without manually creating a virtual environment; `uv` will handle the dependencies for you:

```bash
uv run automatic_differentiation.py
