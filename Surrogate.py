from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class Surrogate:
    
    def __init__(self):
        self.kernel = None
        self.model = None

    def fit(self, X, y):
        self.kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)