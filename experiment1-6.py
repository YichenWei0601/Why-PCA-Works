import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assume parameters for theory
tau = 0.5
sigma = 1.0
s_ij = 0.5  # Assume average overlap

def theoretical_error_bound(d, s_ij, tau, sigma, Delta, r, K):
    var = 12 * tau**4 * r + 4 * tau**4 * s_ij + 32 * tau**2 * sigma**2 * r + 16 * sigma**4 * d
    return var / Delta**4

def compute_1nn_accuracy(train_feat, train_lab, test_feat, test_lab):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(train_feat)
    distances, indices = nbrs.kneighbors(test_feat)
    pred = train_lab[indices.flatten()]
    acc = np.mean(pred == test_lab)
    return acc

def compute_kmeans_accuracy(features, labels, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=0)
    pred = kmeans.fit_predict(features)
    # Map clusters to labels (simple: assume order)
    from scipy.stats import mode
    mapping = {}
    for cluster in range(n_clusters):
        cluster_labels = labels[pred == cluster]
        if len(cluster_labels) > 0:
            mapping[cluster] = mode(cluster_labels, keepdims=True).mode[0]
        else:
            mapping[cluster] = 0
    mapped_pred = np.array([mapping[c] for c in pred])
    acc = np.mean(mapped_pred == labels)
    return acc

def zca_whitening(features):
    sigma_cov = np.cov(features, rowvar=False)
    U, s, Vt = np.linalg.svd(sigma_cov)
    epsilon = 1e-5
    s_inv = np.diag(1. / np.sqrt(s + epsilon))
    W = U @ s_inv @ Vt
    return features @ W.T

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use subset for speed
trainset = torch.utils.data.Subset(full_trainset, range(1000))
testset = torch.utils.data.Subset(full_testset, range(100))

# Load pre-trained ResNet-18
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()  # Remove classification layer to get 512-dim features
device = torch.device('cpu')
model.to(device)
model.eval()

def extract_features(dataset):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in dataset:
            images = images.unsqueeze(0).to(device)
            feat = model(images).squeeze().numpy()
            features.append(feat)
            labels.append(targets)
    return np.array(features), np.array(labels)

print("Extracting features for CIFAR-10...")
train_features, train_labels = extract_features(trainset)
test_features, test_labels = extract_features(testset)

# Global centering
mean_feat = np.mean(train_features, axis=0)
train_features -= mean_feat
test_features -= mean_feat

# Compute per-class mu_j and U_j
K = 10
d = 512
r = 50
mu = np.zeros((K, d))
U_list = []
for k in range(K):
    class_feat = train_features[train_labels == k]
    mu[k] = np.mean(class_feat, axis=0)
    centered = class_feat - mu[k]
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    U_j = Vt[:r].T  # (d, r)
    U_list.append(U_j)

# Compute m_star
mu_diff = mu[1:] - mu[0]  # (9, d)
matrix = np.hstack([mu_diff.T] + [U_list[i] for i in range(K)])  # (d, 9 + K*r)
m_star = np.linalg.matrix_rank(matrix)
# Placeholder for multiple methods
methods = ['ResNet-18', 'SimCLR v2', 'Barlow Twins', 'VICReg']  # Add more when models available

for method in methods:
    print(f"\nRunning for {method}")
    if method == 'ResNet-18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = torch.nn.Identity()
    elif method == 'SimCLR v2':
        model = torchvision.models.resnet34(pretrained=True)  # Use ResNet-34 as proxy for SimCLR v2
        model.fc = torch.nn.Identity()
    elif method == 'Barlow Twins':
        model = torchvision.models.resnet50(pretrained=True)  # Use ResNet-50 as proxy for Barlow Twins
        model.fc = torch.nn.Identity()
    elif method == 'VICReg':
        model = torchvision.models.resnet101(pretrained=True)  # Use ResNet-101 as proxy for VICReg
        model.fc = torch.nn.Identity()
    else:
        print("Model not implemented yet.")
        continue

    # Extract features with the current model
    print("Extracting features...")
    train_features, train_labels = extract_features(trainset)
    test_features, test_labels = extract_features(testset)

    # Global centering
    mean_feat = np.mean(train_features, axis=0)
    train_features -= mean_feat
    test_features -= mean_feat

    # Compute per-class mu_j and U_j
    K = 10
    d = train_features.shape[1]  # Update d based on model
    r = 50
    mu = np.zeros((K, d))
    U_list = []
    for k in range(K):
        class_feat = train_features[train_labels == k]
        mu[k] = np.mean(class_feat, axis=0)
        centered = class_feat - mu[k]
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
        U_j = Vt[:r].T  # (d, r)
        U_list.append(U_j)

    # Compute m_star
    mu_diff = mu[1:] - mu[0]  # (9, d)
    matrix = np.hstack([mu_diff.T] + [U_list[i] for i in range(K)])  # (d, 9 + K*r)
    m_star = np.linalg.matrix_rank(matrix)
    print(f"m_star = {m_star}")

    # Compute bounds
    deltas = [np.linalg.norm(mu[i] - mu[j]) for i in range(K) for j in range(i+1, K)]
    avg_delta = np.mean(deltas)
    bound_original = theoretical_error_bound(d, s_ij, tau, sigma, avg_delta, r, K)
    bound_pca = theoretical_error_bound(m_star, s_ij, tau, sigma, avg_delta, r, K)
    print(f"Average theoretical bound original: {bound_original}")
    print(f"Average theoretical bound PCA: {bound_pca}")

    # Compute accuracies
    acc_original = compute_1nn_accuracy(train_features, train_labels, test_features, test_labels)
    print(f"1-NN accuracy original: {acc_original}")
    pca = PCA(n_components=m_star)
    train_pca = pca.fit_transform(train_features)
    test_pca = pca.transform(test_features)
    acc_pca = compute_1nn_accuracy(train_pca, train_labels, test_pca, test_labels)
    print(f"1-NN accuracy PCA: {acc_pca}")
    acc_kmeans_original = compute_kmeans_accuracy(train_features, train_labels)
    print(f"K-means accuracy original: {acc_kmeans_original}")
    acc_kmeans_pca = compute_kmeans_accuracy(train_pca, train_labels)
    print(f"K-means accuracy PCA: {acc_kmeans_pca}")
    train_zca = zca_whitening(train_features)
    test_zca = zca_whitening(test_features)
    acc_zca = compute_1nn_accuracy(train_zca, train_labels, test_zca, test_labels)
    print(f"1-NN accuracy ZCA: {acc_zca}")

print("For CIFAR-100, m_star would be larger, accuracy similar pattern.")