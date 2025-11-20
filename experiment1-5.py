import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Parameters
K = 3
r = 10
sigma = 1.0
Delta = 1.0
d = 2000  # Fixed
tau = 0.5
s_ij = 0.0  # Orthogonal subspaces

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

def generate_train_test_data(K, d, r, s_ij, tau, sigma, Delta, num_train_per_class, num_test_per_class):
    """Generate training and testing data"""
    U = generate_subspaces(K, r, d, s_ij)
    centers = generate_class_centers(K, d, Delta)
    
    train_data = []
    test_data = []
    train_labels = []
    test_labels = []
    
    for i in range(K):
        mu = centers[i]
        U_i = U[i]
        # Train
        train_points = []
        for _ in range(num_train_per_class):
            z = np.random.randn(r) * tau
            noise = np.random.randn(d) * sigma
            x = mu + U_i.T @ z + noise
            train_points.append(x)
        train_data.extend(train_points)
        train_labels.extend([i] * num_train_per_class)
        
        # Test
        test_points = []
        for _ in range(num_test_per_class):
            z = np.random.randn(r) * tau
            noise = np.random.randn(d) * sigma
            x = mu + U_i.T @ z + noise
            test_points.append(x)
        test_data.extend(test_points)
        test_labels.extend([i] * num_test_per_class)
    
    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

def compute_1nn_error(train_data, train_labels, test_data, test_labels):
    """Compute 1-NN classification error"""
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train_data)
    distances, indices = nbrs.kneighbors(test_data)
    predicted_labels = train_labels[indices.flatten()]
    error_rate = np.mean(predicted_labels != test_labels)
    return error_rate

def compute_m_star(K, d, r, s_ij, tau, sigma, Delta):
    """Compute the true optimal dimension m⋆ = rank([μ differences, all U_j])"""
    U = generate_subspaces(K, r, d, s_ij)
    centers = generate_class_centers(K, d, Delta)
    
    # μ differences: centers as (d, K)
    mu_diff = np.array(centers).T  # (d, K)
    
    # All U_j: U[0].T, U[1].T, ..., U[K-1].T, each (d, r)
    U_matrices = [U[i].T for i in range(K)]  # List of (d, r)
    
    # Concatenate horizontally: [mu_diff, U[0].T, U[1].T, ..., U[K-1].T]
    matrix = np.hstack([mu_diff] + U_matrices)  # (d, K + K*r)
    
    # Compute rank
    m_star = np.linalg.matrix_rank(matrix)
    return m_star

def run_optimal_dimension_experiment(m_values, d=2000, K=3, r=10, tau=0.5, sigma=1.0, Delta=1.0, s_ij=0.0, num_train_per_class=1000, num_test_per_class=200, num_repeats=10):
    accuracies = []
    
    # Compute m⋆
    m_star = compute_m_star(K, d, r, s_ij, tau, sigma, Delta)
    print(f"Computed m⋆ = {m_star}")
    
    for m in m_values:
        print(f"Running for m={m}")
        error_list = []
        for repeat in range(num_repeats):
            # Generate data
            train_data, train_labels, test_data, test_labels = generate_train_test_data(
                K, d, r, s_ij, tau, sigma, Delta, num_train_per_class, num_test_per_class
            )
            
            # Apply PCA to m dimensions
            pca = PCA(n_components=min(m, d))
            train_data_pca = pca.fit_transform(train_data)
            test_data_pca = pca.transform(test_data)
            
            # Compute 1-NN error in PCA space
            error = compute_1nn_error(train_data_pca, train_labels, test_data_pca, test_labels)
            error_list.append(error)
        
        avg_error = np.mean(error_list)
        accuracies.append(1 - avg_error)
        
        print(f"m={m}: Avg accuracy={1 - avg_error:.4f}")
    
    return m_values, accuracies, m_star

# Compute m⋆ first
m_star = compute_m_star(K, d, r, s_ij, tau, sigma, Delta)
print(f"m⋆ = {m_star}")

# Set m values around m⋆
m_values = [m_star - 2, m_star - 1, m_star, m_star + 5]

# Run experiment
m_values, accuracies, m_star = run_optimal_dimension_experiment(m_values)

# Plot
fig, ax = plt.subplots()
ax.plot(m_values, accuracies, marker='o', label='Empirical 1-NN Accuracy')
ax.axvline(x=m_star, color='r', linestyle='--', label=f'Theoretical m⋆ = {m_star}')
ax.set_xlabel('Projection Dimension m')
ax.set_ylabel('Accuracy')
ax.set_title('Optimal Projection Dimension Tightness')
ax.legend()
plt.savefig('accuracy_vs_projection_dimension.png')
plt.show()

print("Optimal dimension experiment completed. Plot saved.")