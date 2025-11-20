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
Delta = 10.0

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

def run_classification_experiment(d_values, K=3, r=10, s_ij=0.5, tau=0.5, sigma=1.0, Delta=10.0, num_train_per_class=500, num_test_per_class=100, num_repeats=5):
    m_opt = (K - 1) + K * r  # 32
    print(f"Optimal dimension m = {m_opt}")
    
    original_errors = []
    pca_errors = []
    random_errors = []
    
    for d in d_values:
        print(f"Running classification for d={d}")
        orig_list = []
        pca_list = []
        rand_list = []
        for repeat in range(num_repeats):
            # Generate data
            train_data, train_labels, test_data, test_labels = generate_train_test_data(
                K, d, r, s_ij, tau, sigma, Delta, num_train_per_class, num_test_per_class
            )
            
            # Original space
            error_orig = compute_1nn_error(train_data, train_labels, test_data, test_labels)
            orig_list.append(error_orig)
            
            # PCA to m_opt
            pca = PCA(n_components=min(m_opt, d))  # Ensure not more than d
            train_pca = pca.fit_transform(train_data)
            test_pca = pca.transform(test_data)
            error_pca = compute_1nn_error(train_pca, train_labels, test_pca, test_labels)
            pca_list.append(error_pca)
            
            # Random projection to m_opt
            if m_opt < d:
                random_proj = np.random.randn(m_opt, d)
                train_rand = train_data @ random_proj.T
                test_rand = test_data @ random_proj.T
            else:
                train_rand = train_data
                test_rand = test_data
            error_rand = compute_1nn_error(train_rand, train_labels, test_rand, test_labels)
            rand_list.append(error_rand)
        
        original_errors.append(np.mean(orig_list))
        pca_errors.append(np.mean(pca_list))
        random_errors.append(np.mean(rand_list))
        
        print(f"d={d}: Original error={np.mean(orig_list):.4f}, PCA error={np.mean(pca_list):.4f}, Random error={np.mean(rand_list):.4f}")
    
    return original_errors, pca_errors, random_errors

# Run classification experiment
d_values_class = [100, 500, 2000]
original_errors, pca_errors, random_errors = run_classification_experiment(d_values_class, num_train_per_class=500, num_test_per_class=100, num_repeats=5)

# Plot
fig, ax = plt.subplots()
ax.plot(d_values_class, original_errors, label='Original space', marker='o')
ax.plot(d_values_class, pca_errors, label='PCA to m=32', marker='x')
ax.plot(d_values_class, random_errors, label='Random projection to 32', marker='s')
ax.set_xlabel('d')
ax.set_ylabel('1-NN Error Rate')
ax.set_title('Dimensionality Explosion and Reduction Benefits')
ax.legend()
ax.set_xscale('log')
plt.savefig('classification_errors_vs_d.png')
plt.show()

print("Classification experiments completed. Plot saved.")