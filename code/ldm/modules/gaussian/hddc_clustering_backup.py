import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

# def GumbelSoftmax(logits, tau=1, normalize=True):
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#     y = logits + gumbel_noise
#     y = torch.nn.functional.softmax(y / tau, dim=-1)
#     if normalize:
#         y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        
#     del gumbel_noise
        
#     return y

class HDDC:
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

    def fit(self, X, eigenvalue_ratio_threshold=10.0, merge_threshold=10.0, max_split_iter=10):
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

        self.split_large_density_clusters(X, eigenvalue_ratio_threshold=eigenvalue_ratio_threshold, max_split_iter=max_split_iter)
        self.merge_clusters(X, merge_threshold=merge_threshold)

    def _e_step(self, X):
        log_likelihoods = []
        eye_matrix = torch.eye(self.covariances[0].shape[0], device=self.device)
        for i in range(self.n_clusters):
            cov_matrix = self.covariances[i] + 1e-6 * eye_matrix
            dist = MultivariateNormal(self.means[i], cov_matrix)
            log_likelihoods.append(dist.log_prob(X))

        log_likelihoods = torch.stack(log_likelihoods).t()
        log_resp = log_likelihoods + torch.log(self.weights.unsqueeze(0))
        log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
        return torch.exp(log_resp)


    def _m_step(self, X, resp):
        Nk = resp.sum(dim=0)
        self.means = torch.mm(resp.t(), X) / Nk.unsqueeze(1)
        
        eye_matrix = torch.eye(X.shape[1], device=self.device)
        for k in range(self.n_clusters):
            diff = X - self.means[k]
            cov_matrix = torch.mm((resp[:, k].unsqueeze(1) * diff).t(), diff) / Nk[k]
            cov_matrix += 1e-6 * eye_matrix
            self.covariances[k] = cov_matrix
        
        self.weights = Nk / Nk.sum()


    def _update_subspaces(self):
        for k in range(self.n_clusters):
            eigvals, eigvecs = torch.linalg.eigh(self.covariances[k] + 1e-6 * torch.eye(self.covariances[k].shape[0], device=self.device))
            self.eigenvalues[k] = eigvals
            self.eigenvectors[k] = eigvecs
            self.dims[k] = (eigvals > self.tol).sum().item()

    def predict0(self, X):
        X = X.to(self.device)
        resp = self._e_step(X)
        return resp.argmax(dim=1)

    def predict(self, X, tau=1.0, lambda_mix=1.0):
        X = X.to(self.device)
        resp = self._e_step(X)
        return lambda_mix * X + (1 - lambda_mix) * GumbelSoftmax(resp, tau=tau) @ self.means

    def covariance_condition_numbers(self):
        return [torch.linalg.cond(self.covariances[i]).item() for i in range(self.n_clusters)]

    def split_large_density_clusters(self, X, eigenvalue_ratio_threshold=10.0, max_split_iter=10):
        split_iter = 0
        while split_iter < max_split_iter:
            if self.max_clusters and self.n_clusters >= self.max_clusters:
                break

            eigenvalue_ratios = [
                (torch.linalg.eigh(self.covariances[i] + 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device))[0][-1] /
                torch.linalg.eigh(self.covariances[i] + 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device))[0][-2])
                if self.covariances[i].shape[0] > 1 else 0.0
                for i in range(self.n_clusters)
            ]

            split_clusters = [i for i, ratio in enumerate(eigenvalue_ratios) if ratio > eigenvalue_ratio_threshold]

            if not split_clusters:
                break

            for i in split_clusters:
                resp = self._e_step(X)
                cluster_data = X[resp[:, i] > 0.5]

                if cluster_data.shape[0] < 20:
                    continue

                n_samples, n_features = cluster_data.shape
                new_indices = torch.randperm(n_samples)[:2]

                new_means = cluster_data[new_indices].to(self.device)
                new_covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(2)])
                new_weights = torch.tensor([0.5, 0.5], device=self.device)

                split_model = HDDC(n_clusters=2, max_iter=self.max_iter, tol=self.tol, device=self.device)
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

                self.means = torch.cat([self.means, split_model.means], dim=0)
                self.covariances = torch.cat([self.covariances, split_model.covariances], dim=0)
                self.weights = torch.cat([self.weights, split_model.weights / 2], dim=0)
                self.n_clusters += 2

                # Remove the original cluster
                self.means = torch.cat([self.means[:i], self.means[i+1:]], dim=0)
                self.covariances = torch.cat([self.covariances[:i], self.covariances[i+1:]], dim=0)
                self.weights = torch.cat([self.weights[:i], self.weights[i+1:]], dim=0)
                self.n_clusters -= 1

                self.weights /= self.weights.sum()

            eigenvalue_ratios = [
                (torch.linalg.eigh(self.covariances[i] + 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device))[0][-1] /
                torch.linalg.eigh(self.covariances[i] + 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device))[0][-2])
                if self.covariances[i].shape[0] > 1 else 0.0
                for i in range(self.n_clusters)
            ]

            if all(ratio <= eigenvalue_ratio_threshold for ratio in eigenvalue_ratios):
                break
            
            split_iter += 1



    def merge_clusters(self, X, merge_threshold=10.0):
        merge_pairs = []
        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                mahalanobis_distance = self.calculate_mahalanobis_distance(self.means[i], self.covariances[i], self.means[j])
                if mahalanobis_distance < merge_threshold:
                    merge_pairs.append((i, j))

        for i, j in merge_pairs:
            if j >= self.n_clusters:
                continue
            self.merge_two_clusters(i, j)

    def calculate_mahalanobis_distance(self, mean1, cov1, mean2):
        diff = mean1 - mean2
        cov_inv = torch.linalg.inv(cov1 + 1e-6 * torch.eye(cov1.shape[0], device=cov1.device))
        return torch.sqrt(torch.dot(torch.matmul(diff, cov_inv), diff))

    def merge_two_clusters(self, i, j):
        Nk_i = self.weights[i]
        Nk_j = self.weights[j]

        new_mean = (Nk_i * self.means[i] + Nk_j * self.means[j]) / (Nk_i + Nk_j)
        new_cov = (Nk_i * self.covariances[i] + Nk_j * self.covariances[j]) / (Nk_i + Nk_j)
        new_weight = Nk_i + Nk_j

        self.means[i] = new_mean
        self.covariances[i] = new_cov
        self.weights[i] = new_weight

        self.means = torch.cat((self.means[:j], self.means[j+1:]), dim=0)
        self.covariances = torch.cat((self.covariances[:j], self.covariances[j+1:]), dim=0)
        self.weights = torch.cat((self.weights[:j], self.weights[j+1:]), dim=0)

        self.n_clusters -= 1
        self.weights /= self.weights.sum()

    def square_mahalanobis_distance(self, mu, sigma, D):
        K, _ = mu.shape
        diff = mu.unsqueeze(1) - mu.unsqueeze(0)
        sigma_inv = torch.linalg.inv(sigma + torch.eye(D, device=sigma.device) * 1e-6)
        mahalanobis_distances = torch.einsum('aij, ajk, aik -> ai', diff, sigma_inv, diff)
        return mahalanobis_distances

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.distributions.multivariate_normal import MultivariateNormal

# class HDDC2:
#     def __init__(self, n_clusters, max_clusters=None, max_iter=100, tol=1e-4, device='cpu', batch_size=10000):
#         self.n_clusters = n_clusters
#         self.max_clusters = max_clusters
#         self.max_iter = max_iter
#         self.tol = tol
#         self.device = device
#         self.batch_size = batch_size
#         self.means = None
#         self.covariances = None
#         self.weights = None
#         self.eigenvalues = None
#         self.eigenvectors = None
#         self.dims = None

#     def fit(self, X, threshold_density=0.5):
#         X = X.to(self.device)
#         n_samples, n_features = X.shape
#         indices = torch.randperm(n_samples)[:self.n_clusters]
#         self.means = X[indices].to(self.device)
#         self.covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(self.n_clusters)])
#         self.weights = torch.ones(self.n_clusters, device=self.device) / self.n_clusters
#         self.eigenvalues = torch.zeros((self.n_clusters, n_features), device=self.device)
#         self.eigenvectors = torch.zeros((self.n_clusters, n_features, n_features), device=self.device)
#         self.dims = torch.zeros(self.n_clusters, dtype=torch.int, device=self.device)

#         for _ in range(self.max_iter):
#             resp = self._e_step(X)
#             self._m_step(X, resp)
#             self._update_subspaces()

#         self.split_large_density_clusters(X, threshold_density=threshold_density)

#     def _e_step(self, X):
#         log_likelihoods = []
#         for i in range(self.n_clusters):
#             likelihoods = []
#             for j in range(0, X.size(0), self.batch_size):
#                 batch = X[j:j+self.batch_size]
#                 if batch.shape[0] == 0:
#                     continue
#                 cov_matrix = self.covariances[i] + 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device)
#                 cov_matrix += 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device)
#                 likelihood = MultivariateNormal(self.means[i], cov_matrix).log_prob(batch)
#                 likelihoods.append(likelihood)
#             if likelihoods:  # Check if likelihoods list is not empty
#                 log_likelihoods.append(torch.cat(likelihoods))
#         log_likelihoods = torch.stack(log_likelihoods)
#         log_resp = log_likelihoods + torch.log(self.weights.unsqueeze(1))
#         log_resp = log_resp - torch.logsumexp(log_resp, dim=0)
#         return torch.exp(log_resp).t()

#     def _m_step(self, X, resp):
#         Nk = resp.sum(dim=0)
#         self.means = torch.mm(resp.t(), X) / Nk.unsqueeze(1)
#         for k in range(self.n_clusters):
#             diff = X - self.means[k]
#             self.covariances[k] = torch.mm(resp[:, k] * diff.t(), diff) / Nk[k]
#         self.weights = Nk / Nk.sum()

#     def _update_subspaces(self):
#         for k in range(self.n_clusters):
#             try:
#                 eigvals, eigvecs = torch.linalg.eigh(self.covariances[k] + 1e-6 * torch.eye(self.covariances[k].shape[0], device=self.device))
#             except RuntimeError:
#                 eigvals, eigvecs = torch.linalg.eigh(self.covariances[k] + 1e-5 * torch.eye(self.covariances[k].shape[0], device=self.device))
#             self.eigenvalues[k] = eigvals
#             self.eigenvectors[k] = eigvecs
#             self.dims[k] = (eigvals > self.tol).sum().item()

#     def predict0(self, X):
#         X = X.to(self.device)
#         resp = self._e_step(X)
#         return resp.argmax(dim=1)

#     def predict(self, X, tau=1e-4):
#         X = X.to(self.device)
#         resp = self._e_step(X)
#         return GumbelSoftmax(resp, tau=tau) @ self.means

#     def covariance_condition_numbers(self):
#         return [torch.linalg.cond(self.covariances[i]).item() for i in range(self.n_clusters)]

#     def split_large_density_clusters(self, X, threshold_density=0.5, max_split_iter=3):
#         split_iter = 0
#         while split_iter < max_split_iter:
#             if self.max_clusters and self.n_clusters >= self.max_clusters:
#                 break
            
#             density_measures = self.evaluate_cluster_density(X)
#             split_clusters = [i for i, density in enumerate(density_measures) if density > threshold_density or torch.rand(1) < 0.5]

#             if not split_clusters:
#                 break

#             for i in split_clusters:
#                 # Get the data points belonging to the split cluster
#                 resp = self._e_step(X)
#                 cluster_data = X[resp[:, i] > 0.5]

#                 if cluster_data.shape[0] < 2:  # Skip if there are not enough data points to split
#                     continue

#                 n_samples, n_features = cluster_data.shape
#                 new_indices = torch.randperm(n_samples)[:2]

#                 new_means = cluster_data[new_indices].to(self.device)
#                 new_covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(2)])
#                 new_weights = torch.tensor([0.5, 0.5], device=self.device)

#                 # Initialize new model for the split cluster
#                 split_model = HDDC(n_clusters=2, max_iter=self.max_iter, tol=self.tol, device=self.device)
#                 split_model.means = new_means
#                 split_model.covariances = new_covariances
#                 split_model.weights = new_weights
#                 split_model.eigenvalues = torch.zeros((2, n_features), device=self.device)
#                 split_model.eigenvectors = torch.zeros((2, n_features, n_features), device=self.device)
#                 split_model.dims = torch.zeros(2, dtype=torch.int, device=self.device)

#                 # Fit the new model to the split cluster data
#                 for _ in range(self.max_iter):
#                     split_resp = split_model._e_step(cluster_data)
#                     split_model._m_step(cluster_data, split_resp)
#                     split_model._update_subspaces()

#                 # Add new clusters to the original model
#                 self.means = torch.cat((self.means, split_model.means), dim=0)
#                 self.covariances = torch.cat((self.covariances, split_model.covariances), dim=0)
#                 self.weights = torch.cat((self.weights, split_model.weights / 2), dim=0)
#                 self.n_clusters += 2

#                 # Remove the old cluster
#                 self.means = torch.cat((self.means[:i], self.means[i+1:]), dim=0)
#                 self.covariances = torch.cat((self.covariances[:i], self.covariances[i+1:]), dim=0)
#                 self.weights = torch.cat((self.weights[:i], self.weights[i+1:]), dim=0)
#                 self.n_clusters -= 1

#                 # Re-normalize the weights
#                 self.weights /= self.weights.sum()

#                 # Check density measures again
#                 density_measures = self.evaluate_cluster_density(X)
#                 if all(density <= threshold_density for density in density_measures):
#                     break
                
#                 split_iter += 1

#     def evaluate_cluster_density(self, X):
#         # Implement a method to evaluate density (e.g., using Mahalanobis distance or other measures)
#         density_measures = []
#         for i in range(self.n_clusters):
#             resp = self._e_step(X)
#             cluster_data = X[resp[:, i] > 0.5]
#             # Example: Calculate Mahalanobis distance or other density measures
#             density = self.calculate_density_measure(cluster_data)  # Implement this function
#             density_measures.append(density)
#         return density_measures

#     def calculate_density_measure(self, cluster_data):
#         # Example function to calculate density measure (e.g., Mahalanobis distance)
#         # Replace with appropriate density calculation based on your problem domain
#         return torch.norm(cluster_data - torch.mean(cluster_data, dim=0), dim=1).mean()

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

class HDDC2:
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

        # 클러스터의 원소 수를 계산하고 작은 클러스터 제거
        resp = self._e_step(X)
        cluster_sizes = resp.sum(dim=0)
        large_clusters = cluster_sizes >= min_cluster_size

        self.means = self.means[large_clusters]
        self.covariances = self.covariances[large_clusters]
        self.weights = self.weights[large_clusters]
        self.weights /= self.weights.sum()  # 가중치 재조정
        self.eigenvalues = self.eigenvalues[large_clusters]
        self.eigenvectors = self.eigenvectors[large_clusters]
        self.dims = self.dims[large_clusters]
        self.n_clusters = large_clusters.sum().item()

        for _ in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            self._update_subspaces()

    # def _e_step(self, X):
    #     log_likelihoods = []
    #     for i in range(self.n_clusters):
    #         likelihoods = []
    #         for j in range(0, X.size(0), self.batch_size):
    #             batch = X[j:j+self.batch_size]
    #             if batch.shape[0] == 0:
    #                 continue
    #             cov_matrix = self.covariances[i] + 1e-6 * torch.eye(self.covariances[i].shape[0], device=self.device)
    #             likelihood = MultivariateNormal(self.means[i], cov_matrix).log_prob(batch)
    #             likelihoods.append(likelihood)
    #         if likelihoods:  # Check if likelihoods list is not empty
    #             log_likelihoods.append(torch.cat(likelihoods))
    #     log_likelihoods = torch.stack(log_likelihoods)
    #     log_resp = log_likelihoods + torch.log(self.weights.unsqueeze(1))
    #     log_resp = log_resp - torch.logsumexp(log_resp, dim=0)
    #     return torch.exp(log_resp).t()

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
                print(f"Error in creating MultivariateNormal for cluster {i}: {e}")
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
                # Frobenius norm을 사용하여 공분산 행렬을 정규화
                norm_covariance = self.covariances[k] / torch.norm(self.covariances[k], p='fro')
                eigvals, eigvecs = torch.linalg.eigh(norm_covariance + 1e-6 * torch.eye(norm_covariance.shape[0], device=self.device))
            except RuntimeError:
                try:
                    eigvals, eigvecs = torch.linalg.eigh(norm_covariance + 1e-5 * torch.eye(norm_covariance.shape[0], device=self.device))
                except RuntimeError:
                    raise RuntimeError(f"Eigenvalue computation failed for cluster {k} after stabilization attempts.")
            
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
                # Get the data points belonging to the split cluster
                resp = self._e_step(X)
                cluster_data = X[resp[:, i] > 0.5]

                if cluster_data.shape[0] < 2:  # Skip if there are not enough data points to split
                    continue

                n_samples, n_features = cluster_data.shape
                new_indices = torch.randperm(n_samples)[:2]

                new_means = cluster_data[new_indices].to(self.device)
                new_covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(2)])
                new_weights = torch.tensor([0.5, 0.5], device=self.device)

                # Initialize new model for the split cluster
                split_model = HDDC2(n_clusters=2, max_iter=self.max_iter, tol=self.tol, device=self.device)
                split_model.means = new_means
                split_model.covariances = new_covariances
                split_model.weights = new_weights
                split_model.eigenvalues = torch.zeros((2, n_features), device=self.device)
                split_model.eigenvectors = torch.zeros((2, n_features, n_features), device=self.device)
                split_model.dims = torch.zeros(2, dtype=torch.int, device=self.device)

                # Fit the new model to the split cluster data
                for _ in range(self.max_iter):
                    split_resp = split_model._e_step(cluster_data)
                    split_model._m_step(cluster_data, split_resp)
                    split_model._update_subspaces()

                # Add new clusters to the original model
                self.means = torch.cat((self.means, split_model.means), dim=0)
                self.covariances = torch.cat((self.covariances, split_model.covariances), dim=0)
                self.weights = torch.cat((self.weights, split_model.weights / 2), dim=0)
                self.eigenvalues = torch.cat((self.eigenvalues, split_model.eigenvalues), dim=0)
                self.eigenvectors = torch.cat((self.eigenvectors, split_model.eigenvectors), dim=0)
                self.dims = torch.cat((self.dims, split_model.dims), dim=0)
                self.n_clusters += 2

                # Remove the old cluster
                self.means = torch.cat((self.means[:i], self.means[i+1:]), dim=0)
                self.covariances = torch.cat((self.covariances[:i], self.covariances[i+1:]), dim=0)
                self.weights = torch.cat((self.weights[:i], self.weights[i+1:]), dim=0)
                self.eigenvalues = torch.cat((self.eigenvalues[:i], self.eigenvalues[i+1:]), dim=0)
                self.eigenvectors = torch.cat((self.eigenvectors[:i], self.eigenvectors[i+1:]), dim=0)
                self.dims = torch.cat((self.dims[:i], self.dims[i+1:]), dim=0)
                self.n_clusters -= 1

                # Re-normalize the weights
                self.weights /= self.weights.sum()

                # Check density measures again
                density_measures = self.evaluate_cluster_density(X)
                if all(density <= threshold_density for density in density_measures):
                    break
                
                split_iter += 1

    def evaluate_cluster_density(self, X):
        # Implement a method to evaluate density (e.g., using Mahalanobis distance or other measures)
        density_measures = []
        for i in range(self.n_clusters):
            resp = self._e_step(X)
            cluster_data = X[resp[:, i] > 0.5]
            # Example: Calculate Mahalanobis distance or other density measures
            density = self.calculate_density_measure(cluster_data)  # Implement this function
            density_measures.append(density)
        return density_measures

    def calculate_density_measure(self, cluster_data):
        # Example function to calculate density measure (e.g., Mahalanobis distance)
        # Replace with appropriate density calculation based on your problem domain
        return torch.norm(cluster_data - torch.mean(cluster_data, dim=0), dim=1).mean()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.distributions.multivariate_normal import MultivariateNormal

# class HDDC:
#     def __init__(self, n_clusters, max_clusters=None, max_iter=100, tol=1e-4, device='cpu', batch_size=10000):
#         self.n_clusters = n_clusters
#         self.max_clusters = max_clusters
#         self.max_iter = max_iter
#         self.tol = tol
#         self.device = device
#         self.batch_size = batch_size
#         self.means = None
#         self.covariances = None
#         self.weights = None
#         self.eigenvalues = None
#         self.eigenvectors = None
#         self.dims = None

#     def fit(self, X, threshold_density=0.5, merge_threshold=10.0):
#         X = X.to(self.device)
#         n_samples, n_features = X.shape
#         indices = torch.randperm(n_samples)[:self.n_clusters]
#         self.means = X[indices].to(self.device)
#         self.covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(self.n_clusters)])
#         self.weights = torch.ones(self.n_clusters, device=self.device) / self.n_clusters
#         self.eigenvalues = torch.zeros((self.n_clusters, n_features), device=self.device)
#         self.eigenvectors = torch.zeros((self.n_clusters, n_features, n_features), device=self.device)
#         self.dims = torch.zeros(self.n_clusters, dtype=torch.int, device=self.device)

#         for _ in range(self.max_iter):
#             resp = self._e_step(X)
#             self._m_step(X, resp)
#             self._update_subspaces()

#         self.split_large_density_clusters(X, threshold_density=threshold_density)
#         self.merge_clusters(X, merge_threshold=merge_threshold)

#     def _e_step(self, X):
#         log_likelihoods = []
#         cov_matrices = self.covariances + 1e-6 * torch.eye(self.covariances[0].shape[0], device=self.device)
#         for i in range(self.n_clusters):
#             cov_matrix = cov_matrices[i]
#             dist = MultivariateNormal(self.means[i], cov_matrix)
#             log_likelihoods.append(dist.log_prob(X))
        
#         log_likelihoods = torch.stack(log_likelihoods).t()
#         log_resp = log_likelihoods + torch.log(self.weights.unsqueeze(0))
#         log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
#         return torch.exp(log_resp)

#     def _m_step(self, X, resp):
#         Nk = resp.sum(dim=0)
#         self.means = torch.mm(resp.t(), X) / Nk.unsqueeze(1)
#         for k in range(self.n_clusters):
#             diff = X - self.means[k]
#             self.covariances[k] = torch.mm((resp[:, k].unsqueeze(1) * diff).t(), diff) / Nk[k]
#         self.weights = Nk / Nk.sum()

#     def _update_subspaces(self):
#         for k in range(self.n_clusters):
#             eigvals, eigvecs = torch.linalg.eigh(self.covariances[k] + 1e-6 * torch.eye(self.covariances[k].shape[0], device=self.device))
#             self.eigenvalues[k] = eigvals
#             self.eigenvectors[k] = eigvecs
#             self.dims[k] = (eigvals > self.tol).sum().item()

#     def predict(self, X):
#         X = X.to(self.device)
#         resp = self._e_step(X)
#         return resp.argmax(dim=1)

#     def covariance_condition_numbers(self):
#         return [torch.linalg.cond(self.covariances[i]).item() for i in range(self.n_clusters)]

#     def split_large_density_clusters(self, X, threshold_density=0.5):
#         while True:
#             if self.max_clusters and self.n_clusters >= self.max_clusters:
#                 break
            
#             density_measures = self.evaluate_cluster_density(X)
#             split_clusters = [i for i, density in enumerate(density_measures) if density > threshold_density]

#             if not split_clusters:
#                 break

#             for i in split_clusters:
#                 resp = self._e_step(X)
#                 cluster_data = X[resp[:, i] > 0.5]

#                 if cluster_data.shape[0] < 2:
#                     continue

#                 n_samples, n_features = cluster_data.shape
#                 new_indices = torch.randperm(n_samples)[:2]

#                 new_means = cluster_data[new_indices].to(self.device)
#                 new_covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(2)])
#                 new_weights = torch.tensor([0.5, 0.5], device=self.device)

#                 split_model = HDDC(n_clusters=2, max_iter=self.max_iter, tol=self.tol, device=self.device)
#                 split_model.means = new_means
#                 split_model.covariances = new_covariances
#                 split_model.weights = new_weights
#                 split_model.eigenvalues = torch.zeros((2, n_features), device=self.device)
#                 split_model.eigenvectors = torch.zeros((2, n_features, n_features), device=self.device)
#                 split_model.dims = torch.zeros(2, dtype=torch.int, device=self.device)

#                 for _ in range(self.max_iter):
#                     split_resp = split_model._e_step(cluster_data)
#                     split_model._m_step(cluster_data, split_resp)
#                     split_model._update_subspaces()

#                 self.means = torch.cat((self.means, split_model.means), dim=0)
#                 self.covariances = torch.cat((self.covariances, split_model.covariances), dim=0)
#                 self.weights = torch.cat((self.weights, split_model.weights / 2), dim=0)
#                 self.n_clusters += 2

#                 self.means = torch.cat((self.means[:i], self.means[i+1:]), dim=0)
#                 self.covariances = torch.cat((self.covariances[:i], self.covariances[i+1:]), dim=0)
#                 self.weights = torch.cat((self.weights[:i], self.weights[i+1:]), dim=0)
#                 self.n_clusters -= 1

#                 self.weights /= self.weights.sum()

#                 density_measures = self.evaluate_cluster_density(X)
#                 if all(density <= threshold_density for density in density_measures):
#                     break

#     def merge_clusters(self, X, merge_threshold=10.0):
#         merge_pairs = []
#         for i in range(self.n_clusters):
#             for j in range(i + 1, self.n_clusters):
#                 mahalanobis_distance = self.calculate_mahalanobis_distance(self.means[i], self.covariances[i], self.means[j])
#                 if mahalanobis_distance < merge_threshold:
#                     merge_pairs.append((i, j))

#         for i, j in merge_pairs:
#             if j >= self.n_clusters:
#                 continue
#             self.merge_two_clusters(i, j)

#     def calculate_mahalanobis_distance(self, mean1, cov1, mean2):
#         diff = mean1 - mean2
#         cov_inv = torch.inverse(cov1 + 1e-6 * torch.eye(cov1.shape[0], device=cov1.device))
#         return torch.sqrt(torch.dot(torch.matmul(diff, cov_inv), diff))

#     def merge_two_clusters(self, i, j):
#         Nk_i = self.weights[i]
#         Nk_j = self.weights[j]

#         new_mean = (Nk_i * self.means[i] + Nk_j * self.means[j]) / (Nk_i + Nk_j)
#         new_cov = (Nk_i * self.covariances[i] + Nk_j * self.covariances[j]) / (Nk_i + Nk_j)
#         new_weight = Nk_i + Nk_j

#         self.means[i] = new_mean
#         self.covariances[i] = new_cov
#         self.weights[i] = new_weight

#         self.means = torch.cat((self.means[:j], self.means[j+1:]), dim=0)
#         self.covariances = torch.cat((self.covariances[:j], self.covariances[j+1:]), dim=0)
#         self.weights = torch.cat((self.weights[:j], self.weights[j+1:]), dim=0)

#         self.n_clusters -= 1
#         self.weights /= self.weights.sum()

#     def evaluate_cluster_density(self, X):
#         density_measures = []
#         for i in range(self.n_clusters):
#             resp = self._e_step(X)
#             cluster_data = X[resp[:, i] > 0.5]
#             density = self.calculate_density_measure(cluster_data)
#             density_measures.append(density)
#         return density_measures

#     def calculate_density_measure(self, cluster_data):
#         return torch.norm(cluster_data - torch.mean(cluster_data, dim=0), dim=1).mean()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.distributions.multivariate_normal import MultivariateNormal

# def GumbelSoftmax(logits, tau=1, normalize=True):
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
#     y = logits + gumbel_noise
#     y = torch.nn.functional.softmax(y / tau, dim=-1)
#     if normalize:
#         y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        
#     del gumbel_noise
        
#     return y

# def stable_eigh(matrix, epsilon=1e-6):
#     if not torch.allclose(matrix, matrix.T, atol=epsilon):
#         matrix = (matrix + matrix.T) / 2
    
#     try:
#         eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
#     except torch.linalg.LinAlgError as e:
#         print(f"LinAlgError 발생: {e}")
#         identity = torch.eye(matrix.size(0), device=matrix.device)
#         matrix += epsilon * identity
#         eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    
#     return eigenvalues, eigenvectors

# class HDDC:
#     def __init__(self, n_clusters, max_clusters=None, max_iter=100, tol=1e-4, device='cuda', batch_size=10000):
#         self.n_clusters = n_clusters
#         self.max_clusters = max_clusters
#         self.max_iter = max_iter
#         self.tol = tol
#         self.device = device
#         self.batch_size = batch_size
#         self.means = None
#         self.covariances = None
#         self.weights = None
#         self.eigenvalues = None
#         self.eigenvectors = None
#         self.dims = None

#     def fit(self, X, threshold_density=0.5, merge_threshold=10.0):
#         X = X.to(self.device)
#         n_samples, n_features = X.shape
#         indices = torch.randperm(n_samples)[:self.n_clusters]
#         self.means = X[indices].to(self.device)
#         self.covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(self.n_clusters)])
#         self.weights = torch.ones(self.n_clusters, device=self.device) / self.n_clusters
#         self.eigenvalues = torch.zeros((self.n_clusters, n_features), device=self.device)
#         self.eigenvectors = torch.zeros((self.n_clusters, n_features, n_features), device=self.device)
#         self.dims = torch.zeros(self.n_clusters, dtype=torch.int, device=self.device)

#         for _ in range(self.max_iter):
#             resp = self._e_step(X)
#             self._m_step(X, resp)
#             self._update_subspaces()

#         self.split_large_density_clusters(X, threshold_density=threshold_density)
#         self.merge_clusters(X, merge_threshold=merge_threshold)

#     def _e_step(self, X):
#         log_likelihoods = []
#         eye_matrix = torch.eye(self.covariances[0].shape[0], device=self.device) 
#         for i in range(self.n_clusters):
#             cov_matrix = self.covariances[i] + 1e-6 * eye_matrix
#             try:
#                 dist = MultivariateNormal(self.means[i], cov_matrix)
#                 log_likelihoods.append(dist.log_prob(X))
#             except RuntimeError:
#                 log_likelihoods.append(torch.full((X.shape[0],), float('-inf'), device=self.device))
        
#         log_likelihoods = torch.stack(log_likelihoods).t()
#         log_resp = log_likelihoods + torch.log(self.weights.unsqueeze(0))
#         log_resp = log_resp - torch.logsumexp(log_resp, dim=1, keepdim=True)
#         return torch.exp(log_resp)


#     def _m_step(self, X, resp):
#         Nk = resp.sum(dim=0)
#         self.means = torch.mm(resp.t(), X) / Nk.unsqueeze(1)
#         for k in range(self.n_clusters):
#             diff = X - self.means[k]
#             self.covariances[k] = torch.mm((resp[:, k].unsqueeze(1) * diff).t(), diff) / Nk[k]
#         self.weights = Nk / Nk.sum()

#     def _update_subspaces(self):
#         for k in range(self.n_clusters):
#             covariance = self.covariances[k] + 1e-4 * torch.eye(self.covariances[k].shape[0], device=self.device)
#             eigvals, eigvecs = stable_eigh(covariance)

#             self.eigenvalues[k] = eigvals
#             self.eigenvectors[k] = eigvecs
#             self.dims[k] = (eigvals > self.tol).sum().item()
            
#     def predict0(self, X):
#         X = X.to(self.device)
#         resp = self._e_step(X)
#         return resp.argmax(dim=1)

#     def predict(self, X, tau=1.0, lambda_mix=1.0):
#         X = X.to(self.device)
#         resp = self._e_step(X)
#         return lambda_mix * X + (1 - lambda_mix) * GumbelSoftmax(resp, tau=tau) @ self.means

#     def covariance_condition_numbers(self):
#         return [torch.linalg.cond(self.covariances[i]).item() for i in range(self.n_clusters)]

#     def split_large_density_clusters(self, X, threshold_density=0.5):
#         while True:
#             if self.max_clusters and self.n_clusters >= self.max_clusters:
#                 break
            
#             density_measures = self.evaluate_cluster_density(X)
#             split_clusters = [i for i, density in enumerate(density_measures) if density > threshold_density]

#             if not split_clusters:
#                 break

#             for i in split_clusters:
#                 resp = self._e_step(X)
#                 cluster_data = X[resp[:, i] > 0.5]

#                 if cluster_data.shape[0] < 2:
#                     continue

#                 n_samples, n_features = cluster_data.shape
#                 new_indices = torch.randperm(n_samples)[:2]

#                 new_means = cluster_data[new_indices].to(self.device)
#                 new_covariances = torch.stack([torch.eye(n_features, device=self.device) for _ in range(2)])
#                 new_weights = torch.tensor([0.5, 0.5], device=self.device)

#                 split_model = HDDC(n_clusters=2, max_iter=20, tol=self.tol, device=self.device)
#                 split_model.means = new_means
#                 split_model.covariances = new_covariances
#                 split_model.weights = new_weights
#                 split_model.eigenvalues = torch.zeros((2, n_features), device=self.device)
#                 split_model.eigenvectors = torch.zeros((2, n_features, n_features), device=self.device)
#                 split_model.dims = torch.zeros(2, dtype=torch.int, device=self.device)

#                 for _ in range(self.max_iter):
#                     split_resp = split_model._e_step(cluster_data)
#                     split_model._m_step(cluster_data, split_resp)
#                     split_model._update_subspaces()

#                 self.means = torch.cat((self.means, split_model.means), dim=0)
#                 self.covariances = torch.cat((self.covariances, split_model.covariances), dim=0)
#                 self.weights = torch.cat((self.weights, split_model.weights / 2), dim=0)
#                 self.n_clusters += 2

#                 self.means = torch.cat((self.means[:i], self.means[i+1:]), dim=0)
#                 self.covariances = torch.cat((self.covariances[:i], self.covariances[i+1:]), dim=0)
#                 self.weights = torch.cat((self.weights[:i], self.weights[i+1:]), dim=0)
#                 self.n_clusters -= 1

#                 self.weights /= self.weights.sum()

#                 density_measures = self.evaluate_cluster_density(X)
#                 if all(density <= threshold_density for density in density_measures):
#                     break

#     def merge_clusters(self, X, merge_threshold=10.0):
#         merge_pairs = []
#         for i in range(self.n_clusters):
#             for j in range(i + 1, self.n_clusters):
#                 mahalanobis_distance = self.calculate_mahalanobis_distance(self.means[i], self.covariances[i], self.means[j])
#                 if mahalanobis_distance < merge_threshold:
#                     merge_pairs.append((i, j))

#         for i, j in merge_pairs:
#             if j >= self.n_clusters:
#                 continue
#             self.merge_two_clusters(i, j)

#     def calculate_mahalanobis_distance(self, mean1, cov1, mean2):
#         diff = mean1 - mean2
#         cov_inv = torch.inverse(cov1 + 1e-6 * torch.eye(cov1.shape[0], device=cov1.device))
#         return torch.sqrt(torch.dot(torch.matmul(diff, cov_inv), diff))

#     def merge_two_clusters(self, i, j):
#         Nk_i = self.weights[i]
#         Nk_j = self.weights[j]

#         new_mean = (Nk_i * self.means[i] + Nk_j * self.means[j]) / (Nk_i + Nk_j)
#         new_cov = (Nk_i * self.covariances[i] + Nk_j * self.covariances[j]) / (Nk_i + Nk_j)
#         new_weight = Nk_i + Nk_j

#         self.means[i] = new_mean
#         self.covariances[i] = new_cov
#         self.weights[i] = new_weight

#         self.means = torch.cat((self.means[:j], self.means[j+1:]), dim=0)
#         self.covariances = torch.cat((self.covariances[:j], self.covariances[j+1:]), dim=0)
#         self.weights = torch.cat((self.weights[:j], self.weights[j+1:]), dim=0)

#         self.n_clusters -= 1
#         self.weights /= self.weights.sum()

#     def evaluate_cluster_density(self, X):
#         density_measures = []
#         for i in range(self.n_clusters):
#             resp = self._e_step(X)
#             cluster_data = X[resp[:, i] > 0.5]
#             density = self.calculate_density_measure(cluster_data)
#             density_measures.append(density)
#         return density_measures

#     def calculate_density_measure(self, cluster_data):
#         return torch.norm(cluster_data - torch.mean(cluster_data, dim=0), dim=1).mean()
    
