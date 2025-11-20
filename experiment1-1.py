import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools


# Parameters
d_values = [100, 500, 2000, 10000]
s_ij_values = [0, 0.2, 0.5, 0.8]
tau_values = [0, 0.5, 1.0]
K = 3
r = 10
sigma = 1.0
Delta = 1.0  # Fixed separation, can be scanned if needed
num_samples = 10000  # Reduced from 10^6 to 10^4 for faster testing
num_points_per_class = 5000  # For generating data distribution

def generate_subspaces(K, r, d, s_ij):
    """Generate subspaces U_i with controlled overlap s_ij"""
    shared_dim = int(round(s_ij * r))
    private_dim = r - shared_dim
    
    # Generate full orthonormal basis
    full_basis = np.linalg.qr(np.random.randn(d, d))[0].T  # (d, d), orthonormal rows
    
    shared = full_basis[:shared_dim]  # (shared_dim, d)
    
    U = []
    for i in range(K):
        private = np.random.randn(private_dim, d)
        if shared_dim > 0:
            proj = shared.T @ shared  # (d, d)
            private = private - (proj @ private.T).T
        private = np.linalg.qr(private.T)[0].T  # Orthonormalize
        U_i = np.vstack([shared, private])
        U.append(U_i)
    
    return U

def generate_class_centers(K, d, Delta):
    """Generate class centers with fixed separation"""
    centers = []
    for i in range(K):
        mu = np.zeros(d)
        if i > 0:
            mu[i-1] = Delta
        centers.append(mu)
    return centers

def generate_data(K, d, r, tau, sigma, U, centers, num_points):
    """Generate data points for each class"""
    data = []
    for i in range(K):
        mu = centers[i]
        U_i = U[i]
        points = []
        for _ in range(num_points):
            z = np.random.randn(r) * tau
            noise = np.random.randn(d) * sigma
            x = mu + U_i.T @ z + noise
            points.append(x)
        data.append(np.array(points))
    return data

def compute_S_ij(data_i, data_j):
    """Compute S_ij from four independent points: 2 for B (between), 2 for W (within)"""
    # Sample x from i, x' from j for B
    idx_x = np.random.choice(len(data_i))
    idx_xp = np.random.choice(len(data_j))
    x = data_i[idx_x]
    xp = data_j[idx_xp]
    B = np.sum((x - xp)**2)
    
    # Sample y, y' from i for W (within class i)
    idx_y, idx_yp = np.random.choice(len(data_i), 2, replace=False)
    y = data_i[idx_y]
    yp = data_i[idx_yp]
    W = np.sum((y - yp)**2)
    
    S_ij = B - W
    return S_ij

def theoretical_var_S_ij(d, s_ij, tau, sigma, Delta, r, K):
    """Theoretical variance of S_ij from Proposition 4.4"""
    # Var(S_ij) = 12τ^4 r + 4τ^4 s_ij + 32τ^2 σ^2 r + 16σ^4 d
    return 12 * tau**4 * r + 4 * tau**4 * s_ij + 32 * tau**2 * sigma**2 * r + 16 * sigma**4 * d

def run_experiment(d, s_ij, tau):
    print(f"Running experiment for d={d}, s_ij={s_ij}, tau={tau}")
    
    # Generate subspaces and centers
    U = generate_subspaces(K, r, d, s_ij)
    centers = generate_class_centers(K, d, Delta)
    
    # Compute actual s_ij for pair (0,1)
    actual_s_ij = np.trace(U[0].T @ U[1] @ U[1].T @ U[0])
    print(f"Actual s_ij: {actual_s_ij}")
    
    # Generate data
    data = generate_data(K, d, r, tau, sigma, U, centers, num_points_per_class)
    
    # Collect S_ij samples for pair (0,1)
    S_ij_samples = []
    for _ in tqdm(range(num_samples)):
        S_ij = compute_S_ij(data[0], data[1])
        S_ij_samples.append(S_ij)
    
    empirical_var = np.var(S_ij_samples)
    theoretical_var = theoretical_var_S_ij(d, actual_s_ij, tau, sigma, Delta, r, K)
    
    return empirical_var, theoretical_var

# Run experiments (reduced set for speed)
results = {}
selected_combinations = [
    (100, 0.5, 0.5),
    (500, 0.5, 0.5),
    (2000, 0.5, 0.5),
    (100, 0.0, 0.5),
    (100, 0.2, 0.5),
    (100, 0.5, 0.5),
    (100, 0.8, 0.5),
    (100, 0.5, 0.0),
    (100, 0.5, 0.5),
    (100, 0.5, 1.0),
]
for d, s_ij, tau in selected_combinations:
    emp, theo = run_experiment(d, s_ij, tau)
    results[(d, s_ij, tau)] = (emp, theo)

# Plotting
# Plot for scanning d, fix s_ij=0.5, tau=0.5
fig, ax = plt.subplots()
s_ij_fixed = 0.5
tau_fixed = 0.5
d_plot = []
emp_plot = []
theo_plot = []
for d in [100, 500, 2000]:  # Only available d
    if (d, s_ij_fixed, tau_fixed) in results:
        emp, theo = results[(d, s_ij_fixed, tau_fixed)]
        d_plot.append(d)
        emp_plot.append(emp)
        theo_plot.append(theo)

ax.plot(d_plot, emp_plot, label='Empirical Var(S_ij)', marker='o')
ax.plot(d_plot, theo_plot, label='Theoretical Var(S_ij)', marker='x')
ax.set_xlabel('d')
ax.set_ylabel('Var(S_ij)')
ax.set_title(f'Var(S_ij) vs d (s_ij={s_ij_fixed}, tau={tau_fixed})')
ax.legend()
plt.savefig('var_S_ij_vs_d.png')
plt.show()

# Similarly for s_ij, fix d=100, tau=0.5
fig, ax = plt.subplots()
d_fixed = 100
tau_fixed = 0.5
s_ij_plot = []
emp_plot = []
theo_plot = []
for s_ij in s_ij_values:
    if (d_fixed, s_ij, tau_fixed) in results:
        emp, theo = results[(d_fixed, s_ij, tau_fixed)]
        s_ij_plot.append(s_ij)
        emp_plot.append(emp)
        theo_plot.append(theo)

ax.plot(s_ij_plot, emp_plot, label='Empirical Var(S_ij)', marker='o')
ax.plot(s_ij_plot, theo_plot, label='Theoretical Var(S_ij)', marker='x')
ax.set_xlabel('s_ij')
ax.set_ylabel('Var(S_ij)')
ax.set_ylim(1000, 3000)
ax.set_title(f'Var(S_ij) vs s_ij (d={d_fixed}, tau={tau_fixed})')
ax.legend()
plt.savefig('var_S_ij_vs_s_ij.png')
plt.show()

# For tau, fix d=100, s_ij=0.5
fig, ax = plt.subplots()
d_fixed = 100
s_ij_fixed = 0.5
tau_plot = []
emp_plot = []
theo_plot = []
for tau in tau_values:
    if (d_fixed, s_ij_fixed, tau) in results:
        emp, theo = results[(d_fixed, s_ij_fixed, tau)]
        tau_plot.append(tau)
        emp_plot.append(emp)
        theo_plot.append(theo)

ax.plot(tau_plot, emp_plot, label='Empirical Var(S_ij)', marker='o')
ax.plot(tau_plot, theo_plot, label='Theoretical Var(S_ij)', marker='x')
ax.set_xlabel('tau')
ax.set_ylabel('Var(S_ij)')
ax.set_ylim(1000, 3000)
ax.set_title(f'Var(S_ij) vs tau (d={d_fixed}, s_ij={s_ij_fixed})')
ax.legend()
plt.savefig('var_S_ij_vs_tau.png')
plt.show()

print("Experiments completed. Plots saved.")

