import numpy as np
from sklearn.linear_model import Lasso

def k_lasso(X, y, weights, K, max_iter=1000, tol=1e-4):
    """
    K-Lasso feature selection alogrithm

    Param:
        X (numpy.ndarray): feature matrix (n_samples, n_features)
        y (numpy.ndarray): target value (n_samples,)
        weights (numpy.ndarray): distance weight (n_samples,)
        K (int): Selected num
        max_iter (int): LASSO max itertion num
        tol (float): LASSO tol

    return:
        selected_features (list): selected feature list
    """
    n_features = X.shape[1]

    # define the search range for lambda
    lambda_max = np.max(np.abs(X.T @ (weights * y)))  # maxium λ value
    lambda_min = 1e-4 * lambda_max
    lambda_mid = (lambda_max + lambda_min) / 2

    selected_features = []
    for _ in range(100):  # max iteration for 100 times
        # config lasso model
        lasso = Lasso(alpha=lambda_mid, max_iter=max_iter, tol=tol)
        
        # Add the weight
        X_weighted = X * np.sqrt(weights[:, np.newaxis])
        y_weighted = y * np.sqrt(weights)
        
        # Fitting the model
        lasso.fit(X_weighted, y_weighted)
        
        # Check non-zero coef number
        nonzero_indices = np.where(lasso.coef_ != 0)[0]
        nonzero_count = len(nonzero_indices)
        
        if nonzero_count == K:
            selected_features = nonzero_indices
            break
        elif nonzero_count > K:
            lambda_min = lambda_mid  # Increse lambda
        else:
            lambda_max = lambda_mid  # Decres lambda
        
        # Update lambda
        lambda_mid = (lambda_max + lambda_min) / 2

    # If not found K features, return the most cloest result
    if len(selected_features) == 0:
        lasso = Lasso(alpha=lambda_mid, max_iter=max_iter, tol=tol)
        lasso.fit(X_weighted, y_weighted)
        selected_features = np.argsort(np.abs(lasso.coef_))[-K:]

    return selected_features.tolist()


# Test
if __name__ == "__main__":
    # simulated data
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = X[:, 0] * 3 + X[:, 1] * 2 + np.random.rand(100)  # only the first and second features contribute to the model
    weights = np.random.rand(100)  # random

    # Run K-Lasso
    K = 2  # Select the two most important feature
    selected_features = k_lasso(X, y, weights, K)
    print(f"Selected features: {selected_features}")