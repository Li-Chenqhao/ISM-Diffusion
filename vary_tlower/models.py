import numpy as np

from setup import OUForwardProcess

'''
Block polygon GMM
'''
class PolygonGMM2D:
    '''
    Polygon GMM: 2D Gaussian mixture model with rotated compoenents and means lie on a regular polygon.
    Fully analytic pdf / logpdf / score, optimized for repeated evaluation.
    Params:
        radius: radius of means of Gaussian components
        sx: standard deviation scaling of primary direction
        sy: standard deviation scaling of secondary direction
        n_compoents: number of Gaussian components, default 6
        angle_offset: the initial location of rotated components
        dim: dimension of distribution, fixed as 2
        weights: weights of the compoenent, we set as None, which means even weighting
    '''
    def __init__(self, radius = 3.0, sx = 1.0, sy = 0.4**2, n_components = 6, angle_offset = 0.5, dim = 2, weights = None):
        self.n_components = n_components
        self.dim = dim

        if weights is None:
            weights = np.ones(self.n_components)
        self.weights = np.asarray(weights, dtype = float)
        self.weights /= self.weights.sum()

        # Angles: phi, phi + (1/3)pi, phi + (2/3)pi, phi + pi, phi + (4/3)pi, phi + (5/3)pi when n_compoenets = 6
        self.angles = angle_offset * np.pi + np.arange(self.n_components) * (2 * np.pi / self.n_components)

        # Parameters of Gaussian compoenets
        self.means = np.zeros((self.n_components, self.dim))
        self.cov = np.zeros((self.n_components, self.dim, self.dim))
        self.cov_inv = np.zeros_like(self.cov)
        self.logdet = np.zeros(self.n_components)

        for k, theta in enumerate(self.angles):
            self.means[k] = radius * np.array([np.cos(theta), np.sin(theta)])
            self.cov[k] = self._cal_cov(theta, sx, sy)
            self.cov_inv[k] = np.linalg.inv(self.cov[k])
            self.logdet[k] = np.linalg.slogdet(self.cov[k])[1]

        # Gaussian normalization constant
        self.log_norm_const = -0.5 * self.dim * np.log(2 * np.pi)

    @staticmethod
    # Covariance matrix
    def _cal_cov(theta, sx, sy):
        scale = np.array([[sx, 0.0],
                          [0.0, sy]])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s],
                      [s,  c]])
        A = R @ scale
        return A @ A.T

    # logpdf of Gaussian compoenents
    def _component_logpdf(self, x):
        '''
        Compute log p_k(x) for all k
        x: (n, 2)
        return: (n, n_components)
        '''
        x = np.atleast_2d(x)
        diff = x[:, None, :] - self.means[None, :, :]  # (n, n_components, 2)

        quad = np.einsum(
            'nkd,kde,nke->nk',
            diff, self.cov_inv, diff
        )

        return self.log_norm_const - 0.5 * (quad + self.logdet)

    # Calculate pdf/logpdf/score
    def logpdf(self, x):
        '''
        Compute log p(x)
        x: (n, 2)
        '''
        logp_k = self._component_logpdf(x)              # (n, n_components)
        return np.log(np.sum(self.weights * np.exp(logp_k), axis = 1))

    def pdf(self, x):
        '''
        Compute p(x)
        x: (n, 2)
        '''
        return np.exp(self.logpdf(x))

    def score(self, x):
        '''
        Compute exact analytic score ∇ log p(x)
        x: (n, 2)
        '''
        x = np.atleast_2d(x)
        diff = x[:, None, :] - self.means[None, :, :]   # (n, n_components, 2)

        logp_k = self._component_logpdf(x)
        p_k = self.weights * np.exp(logp_k)             # (n, n_components)
        norm = np.sum(p_k, axis = 1, keepdims = True)       # (n, 1)

        # Gaussian scores: -Σ^{-1}(x - μ)
        gauss_scores = -np.einsum(
            'kde,nke->nkd',
            self.cov_inv, diff
        )                                               # (n, n_components, 2)

        return np.sum(p_k[:, :, None] * gauss_scores, axis = 1) / norm

    def rvs(self, n):
        '''
        Sampling from cross GMM distribution
        n: Sample size
        '''
        idx = np.random.choice(self.n_components, size = n, p = self.weights)
        eps = np.random.randn(n, self.dim)
        L = np.linalg.cholesky(self.cov[idx])
        return self.means[idx] + np.einsum('nij,nj->ni', L, eps)

class BlockPolygonGMM:
    '''
    Higher-dimensional blockwise polygon GMM.
    Each block is a 2D polygon GMM, and different blocks are independent.
    params:
        n_blocks: number of blocks, the dimension is 2 * n_blocks
        block_kwargs: parameters from PolygonGMM2D
    '''
    def __init__(self, n_blocks: int, **block_kwargs):
        self.n_blocks = n_blocks
        self.blocks = [PolygonGMM2D(**block_kwargs)
                       for _ in range(n_blocks)]

    # Sampling
    def rvs(self, n):
        samples = [block.rvs(n) for block in self.blocks]
        return np.concatenate(samples, axis = 1)

    # Compute logpdf and score
    def logpdf(self, x):
        x = np.atleast_2d(x)
        logp = 0.0

        for b, block in enumerate(self.blocks):
            xb = x[:, 2*b:2*b+2]
            logp += np.array([block.logpdf(xi) for xi in xb])

        return logp

    def score(self, x):
        x = np.atleast_2d(x)
        scores = []

        for b, block in enumerate(self.blocks):
            xb = x[:, 2*b:2*b+2]
            scores.append(np.array([block.score(xi) for xi in xb]))

        return np.concatenate(scores, axis = 1)

# ---------------------------------------
# Oracle score of polygon GMM in OU process
# ---------------------------------------
class PolygonGMM2D_OUScore:
    '''
    True score ∇ log p_t(x) for a 2D polygon GMM under OU forward process.
    GMM after OU process is still GMM.
    p_t(x) = sum_k pi_k varphi(x; m_t μ_k, m_t^2 Σ_k + σ_t^2 I)
    The score is a weighted average of the scores of GMM components.
    '''
    def __init__(self, gmm2d: PolygonGMM2D, diffuse: OUForwardProcess):
        self.gmm2d = gmm2d
        self.diffuse = diffuse
        self.dim = 2
        self.logp_constant = self.dim/2 * np.log(2 * np.pi) - np.log(gmm2d.weights)

    def score(self, x, t):
        '''
        x: (n, 2)
        t: (n, 1)
        return: (n, 2)
        '''
        x = np.atleast_2d(x)
        t = np.asarray(t)

        m = self.diffuse.m_t(t)[:, 0] # (n,)
        sigma2 = self.diffuse.sigma_t(t)[:, 0] ** 2 # (n,)

        # OU-transformed parameters
        means_t = m[:, None, None] * self.gmm2d.means[None, :, :] # (n, K, 2)
        covs_t = (
            (m[:, None, None, None] ** 2) * self.gmm2d.cov[None, :, :, :]
            + sigma2[:, None, None, None] * np.eye(2)
        ) # (n, K, 2, 2)

        cov_inv = np.linalg.inv(covs_t) # (n, K, 2, 2)
        logdet = np.linalg.slogdet(covs_t)[1] # (n, K)

        diff = x[:, None, :] - means_t # (n, K, 2)
        quad = np.einsum('nkd,nkde,nke->nk', diff, cov_inv, diff)

        # log pi_k + log N_k(x)
        logp_k = -(0.5 * quad + 0.5 * logdet + self.logp_constant) # (n, K)

        # Posterior responsibility
        logp_k -= logp_k.max(axis = 1, keepdims = True) # Deducting a common constant for numerical stability
        w = np.exp(logp_k) 
        w /= w.sum(axis = 1, keepdims = True) # (n, K)

        # Gaussian scores
        gauss_scores = -np.einsum('nkde,nke->nkd', cov_inv, diff) # (n, K, 2)

        return np.sum(w[:, :, None] * gauss_scores, axis = 1)

class BlockPolygonGMM_OUScore:
    '''
    True score ∇ log p_t(x) for BlockPolygonGMM under OU forward process.
    '''
    def __init__(self, block_gmm: BlockPolygonGMM, diffuse: OUForwardProcess):
        self.blocks = block_gmm.blocks
        self.n_blocks = block_gmm.n_blocks

        self.block_scores = [
            PolygonGMM2D_OUScore(block, diffuse)
            for block in self.blocks
        ]

    def score(self, x, t):
        """
        x: (n, 2 * n_blocks)
        t: scalar or (n, 1)
        """
        x = np.atleast_2d(x)
        scores = []

        for b, scorer in enumerate(self.block_scores):
            xb = x[:, 2*b:2*b+2]
            scores.append(scorer.score(xb, t))

        return np.concatenate(scores, axis = 1)

'''
Gaussian pushforward
'''
class InvertibleMap:
    '''
    Abstract class of invertible maps T: z ∈ R^d -> x ∈ R^d
    '''
    def __init__(self):
        self.dim = None

    def forward(self, z):
        raise NotImplementedError

    def inverse(self, x):
        raise NotImplementedError

    def log_det_jacobian(self, z):
        """
        log |det ∇T(z)|
        """
        raise NotImplementedError
    
class GaussianPushforward:
    '''
    Gaussian pushforward distribution: X = T(Z), Z ~ N(0, I)
    Analytic pdf / score via change of variables
    Params:
        mapping: an invertible mapping T from InvertibleMap class
    '''
    def __init__(self, mapping: InvertibleMap):
        self.map = mapping
        self.dim = self.map.dim  # inferred from the mapping
        self.log_norm_const = -0.5 * self.dim * np.log(2 * np.pi)

    # Sampling
    def rvs(self, n):
        z = np.random.randn(n, self.dim)
        return self.map.forward(z)

    # Compute logpdf/pdf/score
    def logpdf(self, x):
        z = self.map.inverse(x)
        quad = np.sum(z**2, axis=1)
        log_base = -0.5 * quad
        return self.log_norm_const + log_base - self.map.log_det_jacobian(z)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

class TanhLinearMap(InvertibleMap):
    '''
    A concrete invertible nonlinear map T(z) = A tanh(z)
    Params:
        A: a square and invertible parameter matrix
    '''
    def __init__(self, A):
        A = np.asarray(A)
        assert A.shape[0] == A.shape[1], "A must be square"
        self.A = A
        self.A_inv = np.linalg.inv(A)
        self.dim = A.shape[0]
        self.logdet_A = np.linalg.slogdet(A)[1]

    @staticmethod
    def _arctanh(y):
        '''
        Elementwise inverse function of tanh
        '''
        return 0.5 * np.log((1 + y) / (1 - y))

    def forward(self, z):
        '''
        Forward mapping x = A tanh(z)
        z: (n, d)
        '''
        return np.tanh(z) @ self.A.T

    def inverse(self, x):
        '''
        Inverse mapping z = arctanh(A^{-1} x)
        '''
        y = x @ self.A_inv.T
        return self._arctanh(y)

    def log_det_jacobian(self, z):
        '''
        log |det ∇T(z)| = log|det A| + sum log(1 - tanh^2(z_i))
        '''
        log_det_nonlinear = np.sum(np.log(1 - np.tanh(z)**2), axis=1)
        return self.logdet_A + log_det_nonlinear
    
def get_A_matrix(config, simple_lower = 0.6, simple_upper = 1.5, complex_lower = -0.5, complex_upper = 0.5):
    '''
    Construct a linear matrix A used in the tanh linear map, supporting two regimes controlled by config.simple_matrix.
    If config.simple_matrix is True, A is a diagonal matrix with varying positive scales.
    If config.simple_matrix is False, A is constructed as A = Q D Q^T, where Q is a random orthogonal matrix and D has log-uniformly distributed positive eigenvalues.
    Params:
        config: configuration object.
        simple_lower: lower bound for diagonal scaling factors in the simple regime.
        simple_upper: upper bound for diagonal scaling factors in the simple regime.
        complex_lower: lower bound of the log-eigenvalue range in the complex regime.
        complex_upper: upper bound of the log-eigenvalue range in the complex regime.
    '''
    if config.model.simple_matrix:
        scales = np.linspace(simple_lower, simple_upper, config.dim)
        return np.diag(scales)
    else:
        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(config.dim, config.dim))

        # Anisotropic spectrum
        eigvals = np.exp(np.random.uniform(complex_lower, complex_upper, size = config.dim))
        D = np.diag(eigvals)

        return Q @ D @ Q.T

'''
Energy model
'''
class EnergyModel:
    '''
    Abstract class of energy model p(x) ∝ exp(-E(x))
    '''
    def energy(self, x):
        raise NotImplementedError

    def grad_energy(self, x):
        raise NotImplementedError

    def score(self, x):
        return -self.grad_energy(x)

class QuarticEnergy(EnergyModel):
    '''
    A concrete energy model E(x) = 1/2 x^T Q x + alpha * sum x_i^4
    Params:
        Q: a square matrix, recommended to use a symetric one
        alpha: a positive constant
    '''
    def __init__(self, Q, alpha = 0.1):
        assert Q.shape[0] == Q.shape[1], 'Q must be square'
        assert alpha > 0, 'alpha must be positive'
        self.Q = 0.5 * (Q + Q.T) # symmetrize
        self.alpha = alpha
        self.dim = Q.shape[0]

    def energy(self, x):
        quad = 0.5 * np.einsum('ni,ij,nj->n', x, self.Q, x)
        quartic = self.alpha * np.sum(x**4, axis=1)
        return quad + quartic

    def grad_energy(self, x):
        return x @ self.Q + 4 * self.alpha * x**3

    def rvs(self, n, step = 1e-2, burnin = 200):
        '''
        Langevin sampling
        '''
        x = np.random.randn(n, self.dim)

        for _ in range(burnin):
            x -= step * self.grad_energy(x)
            x += np.sqrt(2 * step) * np.random.randn(*x.shape)

        return x
    
def get_Q_matrix(config, complex_lower = -1.0, complex_upper = 1.0):
    '''
    Construct a symmetric matrix Q defining the quadratic term of quartic energy, supporting two regimes controlled by config.simple_matrix.
    If config.simple_matrix is True, Q is the identity matrix.
    If config.simple_matrix is False, Q is constructed as Q = U D U^T, where U is a random orthogonal matrix and D contains eigenvalues spanning both positive and negative values.
    Params:
        config: configuration object.
        complex_lower: lower bound of the eigenvalue range in the complex regime.
        complex_upper: upper bound of the eigenvalue range in the complex regime.
    '''
    if config.model.simple_matrix:
        return np.eye(config.dim)
    else:
        # Symmetric matrix with mixed eigenvalues
        U, _ = np.linalg.qr(np.random.randn(config.dim, config.dim))
        eigvals = np.linspace(complex_lower, complex_upper, config.dim)
        D = np.diag(eigvals)

        return U @ D @ U.T