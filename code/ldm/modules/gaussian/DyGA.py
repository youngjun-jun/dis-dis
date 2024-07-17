import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

def GumbelSoftmax(logits, tau=1, normalize=True):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    y = torch.nn.functional.softmax(y / tau, dim=-1)
    if normalize:
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        
    return y

class DyGA:
    def __init__(self, n_clusters, max_clusters=None, max_iter=100, tol=1e-4, device='cpu', batch_size=10000):
        self.n_clusters = n_clusters
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.device = device
        self.batch_size = batch_size
        self.means = None
        self.covariances = None
        self.weights = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.dims = None

    def fit(self, X, threshold_density=0.5, min_cluster_size=10):
        X = X.to(self.device)
        n_samples, n_features = X.shape
        indices = torch.randperm(n_samples)[:self.n_clusters]
        self.means = X[indices].to(self.device)
        self.covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(self.n_clusters)])
        self.weights = torch.ones(self.n_clusters, device=self.device) / self.n_clusters
        self.eigenvalues = torch.zeros((self.n_clusters, n_features), device=self.device)
        self.eigenvectors = torch.zeros((self.n_clusters, n_features, n_features), device=self.device)
        self.dims = torch.zeros(self.n_clusters, dtype=torch.int, device=self.device)

        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            self._update_subspaces()

        self.split_large_density_clusters(X, threshold_density=threshold_density)

        resp = self._e_step(X)
        cluster_sizes = resp.sum(dim=0)
        large_clusters = cluster_sizes >= min_cluster_size

        self.means = self.means[large_clusters]
        self.covariances = self.covariances[large_clusters]
        self.weights = self.weights[large_clusters]
        self.weights /= self.weights.sum()
        self.eigenvalues = self.eigenvalues[large_clusters]
        self.eigenvectors = self.eigenvectors[large_clusters]
        self.dims = self.dims[large_clusters]
        self.n_clusters = large_clusters.sum().item()

        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            self._update_subspaces()

    def _e_step(self, X):
        log_likelihoods = []
        eye_matrix = torch.eye(self.covariances[0].shape[0], device=self.device)
        for i in range(self.n_clusters):
            cov_matrix = self.covariances[i] + 1e-6 * eye_matrix
            try:
                dist = MultivariateNormal(self.means[i], cov_matrix)
                log_likelihoods.append(dist.log_prob(X))
                del dist
            except RuntimeError as e:
                log_likelihoods.append(torch.full((X.shape[0],), float('-inf'), device=self.device))

        log_likelihoods = torch.stack(log_likelihoods).t()
        log_resp = log_likelihoods + torch.log(self.weights.unsqueeze(0))
        log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp)

    def _m_step(self, X, resp):
        Nk = resp.sum(dim=0)
        self.means = torch.mm(resp.t(), X) / Nk.unsqueeze(1)
        for k in range(self.n_clusters):
            diff = X - self.means[k]
            self.covariances[k] = torch.mm(resp[:, k] * diff.t(), diff) / Nk[k]
        self.weights = Nk / Nk.sum()

    def _update_subspaces(self):
        for k in range(self.n_clusters):
            try:
                norm_covariance = self.covariances[k] / torch.norm(self.covariances[k], p='fro')
                eigvals, eigvecs = torch.linalg.eigh(norm_covariance + 1e-6 * torch.eye(norm_covariance.shape[0], device=self.device))
            except RuntimeError:
                try:
                    eigvals, eigvecs = torch.linalg.eigh(norm_covariance + 1e-5 * torch.eye(norm_covariance.shape[0], device=self.device))
                except RuntimeError:
                    raise RuntimeError
            
            self.eigenvalues[k] = eigvals
            self.eigenvectors[k] = eigvecs
            self.dims[k] = (eigvals > self.tol).sum().item()


    def predict0(self, X):
        X = X.to(self.device)
        resp = self._e_step(X)
        return self.means[resp.argmax(dim=1)]

    def predict(self, X, tau=1e-4):
        X = X.to(self.device)
        resp = self._e_step(X)
        return GumbelSoftmax(resp, tau=tau) @ self.means

    def covariance_condition_numbers(self):
        return [torch.linalg.cond(self.covariances[i]).item() for i in range(self.n_clusters)]

    def split_large_density_clusters(self, X, threshold_density=0.5, max_split_iter=3):
        split_iter = 0
        while split_iter < max_split_iter:
            if self.max_clusters and self.n_clusters >= self.max_clusters:
                break
            
            density_measures = self.evaluate_cluster_density(X)
            split_clusters = [i for i, density in enumerate(density_measures) if density > threshold_density or torch.rand(1) < 0.5]

            if not split_clusters:
                break

            for i in split_clusters:
                resp = self._e_step(X)
                cluster_data = X[resp[:, i] > 0.5]

                if cluster_data.shape[0] < 2: 
                    continue

                n_samples, n_features = cluster_data.shape
                new_indices = torch.randperm(n_samples)[:2]

                new_means = cluster_data[new_indices].to(self.device)
                new_covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(2)])
                new_weights = torch.tensor([0.5, 0.5], device=self.device)

                split_model = DyGA(n_clusters=2, max_iter=self.max_iter, tol=self.tol, device=self.device)
                split_model.means = new_means
                split_model.covariances = new_covariances
                split_model.weights = new_weights
                split_model.eigenvalues = torch.zeros((2, n_features), device=self.device)
                split_model.eigenvectors = torch.zeros((2, n_features, n_features), device=self.device)
                split_model.dims = torch.zeros(2, dtype=torch.int, device=self.device)

                for _ in range(self.max_iter):
                    split_resp = split_model._e_step(cluster_data)
                    split_model._m_step(cluster_data, split_resp)
                    split_model._update_subspaces()

                self.means = torch.cat((self.means, split_model.means), dim=0)
                self.covariances = torch.cat((self.covariances, split_model.covariances), dim=0)
                self.weights = torch.cat((self.weights, split_model.weights / 2), dim=0)
                self.eigenvalues = torch.cat((self.eigenvalues, split_model.eigenvalues), dim=0)
                self.eigenvectors = torch.cat((self.eigenvectors, split_model.eigenvectors), dim=0)
                self.dims = torch.cat((self.dims, split_model.dims), dim=0)
                self.n_clusters += 2

                self.means = torch.cat((self.means[:i], self.means[i+1:]), dim=0)
                self.covariances = torch.cat((self.covariances[:i], self.covariances[i+1:]), dim=0)
                self.weights = torch.cat((self.weights[:i], self.weights[i+1:]), dim=0)
                self.eigenvalues = torch.cat((self.eigenvalues[:i], self.eigenvalues[i+1:]), dim=0)
                self.eigenvectors = torch.cat((self.eigenvectors[:i], self.eigenvectors[i+1:]), dim=0)
                self.dims = torch.cat((self.dims[:i], self.dims[i+1:]), dim=0)
                self.n_clusters -= 1

                self.weights /= self.weights.sum()

                density_measures = self.evaluate_cluster_density(X)
                if all(density <= threshold_density for density in density_measures):
                    break
                
                split_iter += 1

    def evaluate_cluster_density(self, X):
        density_measures = []
        for i in range(self.n_clusters):
            resp = self._e_step(X)
            cluster_data = X[resp[:, i] > 0.5]
            density = self.calculate_density_measure(cluster_data)
            density_measures.append(density)
        return density_measures

    def calculate_density_measure(self, cluster_data):
        return torch.norm(cluster_data - torch.mean(cluster_data, dim=0), dim=1).mean()
