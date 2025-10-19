import numpy as np
import GPy
from scipy.stats import norm
from sklearn.cluster import KMeans
from GPy.inference.latent_function_inference import ExactGaussianInference

# ------------------ Custom Likelihoods ------------------ #

class ProbitLikelihood(GPy.likelihoods.Likelihood):
    def __init__(self, nu=1e-6, gp_link=None, name="probit"):
        if gp_link is None:
            gp_link = GPy.likelihoods.link_functions.Identity()
        super(ProbitLikelihood, self).__init__(gp_link=gp_link, name=name)
        self.nu = nu

    def pdf(self, F, Y):
        return np.clip(norm.cdf(Y * F), 1e-12, 1.0)

    def logpdf(self, F, Y):
        return np.log(np.clip(norm.cdf(Y * F), 1e-12, 1.0))

    def dlogpdf_dF(self, F, Y):
        pdf_val = norm.pdf(Y * F)
        cdf_val = np.clip(norm.cdf(Y * F), 1e-12, 1.0)
        return Y * (pdf_val / cdf_val)

    def d2logpdf_dF2(self, F, Y):
        d1 = self.dlogpdf_dF(F, Y)
        pdf_val = norm.pdf(Y * F)
        cdf_val = np.clip(norm.cdf(Y * F), 1e-12, 1.0)
        return -d1 * (Y * pdf_val / cdf_val)


class CompositeLikelihood(GPy.likelihoods.Likelihood):
    """Composite likelihood that delegates to a Gaussian likelihood for real observations
    and a Probit likelihood for derivative/monotonicity virtual observations.
    The second column of Y is used as an indicator: 1 => gaussian, 2 => probit.
    """

    def __init__(self, gauss, probit, gp_link=None, name="composite"):
        if gp_link is None:
            gp_link = GPy.likelihoods.link_functions.Identity()
        super(CompositeLikelihood, self).__init__(gp_link=gp_link, name=name)
        self.gauss = gauss
        self.probit = probit

    def logpdf(self, F, Y):
        ll = np.zeros(F.shape)
        for i in range(F.shape[0]):
            if Y[i, 1] == 1:
                ll[i, 0] = self.gauss.logpdf(F[i, 0], Y[i, 0])
            else:
                ll[i, 0] = self.probit.logpdf(F[i, 0], Y[i, 0])
        return ll

    def dlogpdf_dF(self, F, Y):
        dldF = np.zeros(F.shape)
        for i in range(F.shape[0]):
            if Y[i, 1] == 1:
                dldF[i, 0] = self.gauss.dlogpdf_dF(F[i, 0], Y[i, 0])
            else:
                dldF[i, 0] = self.probit.dlogpdf_dF(F[i, 0], Y[i, 0])
        return dldF

    def d2logpdf_dF2(self, F, Y):
        d2ldF2 = np.zeros(F.shape)
        for i in range(F.shape[0]):
            if Y[i, 1] == 1:
                d2ldF2[i, 0] = self.gauss.d2logpdf_dF2(F[i, 0], Y[i, 0])
            else:
                d2ldF2[i, 0] = self.probit.d2logpdf_dF2(F[i, 0], Y[i, 0])
        return d2ldF2

    # delegate other methods to the Gaussian likelihood so GPy
    # internals (predictive mean/variance, sampling, etc.) keep working.
    def gaussian_variance(self, Y_metadata=None):
        return self.gauss.gaussian_variance(Y_metadata)

    def predictive_mean(self, mu, var, Y_metadata=None):
        return self.gauss.predictive_mean(mu, var)

    def predictive_variance(self, mu, var, predictive_mean=None, Y_metadata=None):
        return self.gauss.predictive_variance(mu, var, predictive_mean)

    def samples(self, F, Y_metadata=None, samples=500):
        return self.gauss.samples(F, Y_metadata, samples)

    def pdf_link(self, f, y, Y_metadata=None):
        # convenience method to compute likelihood of scalar f given y (with indicator)
        if np.ndim(y) == 0 or len(y) == 1:
            target = y if np.ndim(y) == 0 else y[0]
            indicator = 1
        else:
            target, indicator = y[0], y[1]

        f_arr = np.array([[f]])
        y_arr = np.array([[target, indicator]])

        if indicator == 1:
            return np.exp(self.gauss.logpdf(f_arr, y_arr)[0, 0])
        elif indicator == 2:
            return np.exp(self.probit.logpdf(f_arr, y_arr)[0, 0])
        else:
            raise ValueError(f"Unknown indicator value: {indicator}")


# ------------------ Helper Functions ------------------ #

def predict_grad(model, X, eps=1e-5):
    """Numerical gradient of the posterior mean wrt inputs X.
    Returns an (n, d) array.
    """
    n, d = X.shape
    grad = np.zeros((n, d))
    mu, _ = model.predict(X)
    for j in range(d):
        X_perturb = X.copy()
        X_perturb[:, j] += eps
        mu_eps, _ = model.predict(X_perturb)
        grad[:, j] = (mu_eps[:, 0] - mu[:, 0]) / eps
    return grad


def setUpDataForMonotonic(gp, X, Y):
    """Create augmented X and Y including virtual derivative observations.
    - Real observations: indicator 1
    - Virtual derivative observations: indicator 2 (probit)
    The derivative virtual inputs are stored in gp.xv and directions in gp.nvd.
    """
    n_true = X.shape[0]
    X_true = np.hstack([X, np.zeros((n_true, 1))])
    Y_true = np.hstack([Y.reshape(-1, 1), np.ones((n_true, 1))])

    xv = gp.xv
    nvd = gp.nvd
    X_deriv_list = []
    Y_deriv_list = []
    for dval in nvd:
        Xd = np.hstack([xv, np.full((xv.shape[0], 1), abs(dval))])
        Yd = np.hstack([np.full((xv.shape[0], 1), np.sign(dval)),
                        2 * np.ones((xv.shape[0], 1))])
        X_deriv_list.append(Xd)
        Y_deriv_list.append(Yd)
    X_deriv = np.vstack(X_deriv_list)
    Y_deriv = np.vstack(Y_deriv_list)

    X_aug = np.vstack([X_true, X_deriv])
    Y_aug = np.vstack([Y_true, Y_deriv])
    return X_aug, Y_aug


def gp_monotonic(gp, X, Y, nv=None, init='sample', xv=None, nvd=None,
                 nu=1e-3, force=True, display=True, optimize='on', opt=None):
    """Construct a monotonic version of `gp` by adding virtual derivative
    observations at points xv (initialized by `init`).

    Returns a new GPy.core.GP instance with CompositeLikelihood and
    (optionally) EP inference when adding virtual points.
    """
    # local import -- EP is only needed if force-loop runs
    from GPy.inference.latent_function_inference.expectation_propagation import EP

    N = X.shape[0]
    if nv is None:
        nv = int(np.floor(0.25 * N))
    if xv is None:
        if init == 'sample':
            perm = np.random.permutation(N)
            xv = X[perm[:nv], :]
        elif init == 'kmeans':
            km = KMeans(n_clusters=nv, random_state=0).fit(X)
            xv = km.cluster_centers_
        else:
            raise ValueError("Unknown init method. Use 'sample' or 'kmeans'.")
    gp.xv = xv

    if nvd is None:
        gp.nvd = np.ones(X.shape[1])
    else:
        gp.nvd = np.array(nvd)

    gauss_like = gp.likelihood
    probit_like = ProbitLikelihood(nu=nu)
    comp_like = CompositeLikelihood(gauss_like, probit_like)

    X_aug, Y_aug = setUpDataForMonotonic(gp, X, Y)
    X_for_gp = X_aug[:, :1]
    Y_for_gp = Y_aug[:, :1]

    gp_new = GPy.core.GP(
        X_for_gp,
        Y_for_gp,
        gp.kern,
        likelihood=comp_like,
        inference_method=ExactGaussianInference()
    )
    gp_new.xv = gp.xv
    gp_new.nvd = gp.nvd
    gp_new.jitter = 1e-6

    if opt is None:
        opt = {'max_iters': 100, 'messages': display}
    if optimize.lower() == 'on':
        if display:
            print("Optimizing hyperparameters (initial) with Exact inference...")
        gp_new.optimize(**opt)

    # If force==True, iteratively add virtual points where monotonicity
    # violations are detected. Use EP as the inference method when re-fitting.
    if force:
        max_loops = 20
        loop_count = 0
        while loop_count < max_loops:
            Ef = predict_grad(gp_new, X)
            sign_vec = np.sign(gp_new.nvd)
            violation_found = False
            worst_indices = []
            for j in range(len(sign_vec)):
                check_vals = Ef[:, j] * sign_vec[j]
                sorted_idx = np.argsort(check_vals)
                picks = []
                for idx in sorted_idx:
                    if check_vals[idx] < -nu:
                        picks.append(idx)
                    if len(picks) == 2:
                        break
                if picks:
                    violation_found = True
                    worst_indices.extend(picks)
            worst_indices = np.unique(worst_indices)
            if not violation_found:
                if display:
                    print("No violations detected. Monotonicity enforced.")
                break

            if display:
                print(f"Iteration {loop_count+1}: Adding {len(worst_indices)} virtual observation(s).")

            # add virtual points to gp.xv
            gp.xv = np.vstack([gp.xv, X[worst_indices]])

            # rebuild augmented data and create a GP with EP inference
            X_aug, Y_aug = setUpDataForMonotonic(gp, X, Y)
            gp_new = GPy.core.GP(X_aug[:, :1], Y_aug[:, :1], gp_new.kern,
                                 likelihood=CompositeLikelihood(gauss_like, probit_like),
                                 inference_method=EP())
            gp_new.xv = gp.xv
            gp_new.nvd = gp.nvd
            gp_new.jitter = 1e-2

            if optimize.lower() == 'on':
                if display:
                    print("Optimizing hyperparameters after adding virtual points with EP...")
                gp_new.optimize(**opt)
            loop_count += 1

        if loop_count == max_loops:
            print("Warning: Maximum iterations reached. Monotonicity may be incomplete.")

        if display:
            print("Final number of virtual points:", gp_new.xv.shape[0])

    return gp_new
