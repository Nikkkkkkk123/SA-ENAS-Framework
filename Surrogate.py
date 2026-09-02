from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class Surrogate:
    
    def __init__(self):
        self.kernel = None
        self.model = None

    def fit(self, X, y):
        self.kernel = ConstantKernel(1.0, (1e-6, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-8, 1e8)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-15, 1e5))
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)