import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
from scipy.stats import pearsonr
import sys, json
from itertools import product
from scipy.linalg import sqrtm




def gen_exo(n, d):
    """
    Generates exogeneous varables; independent normal random variables
    """
    z = np.random.normal(size = (n, d))
    return z

def gen_latent(n, d, rho = 0):
    """
    Generates latent factors, 
    Letent factors: A d dimensional normal random vectors with mean zero 
                    and the first two co-ordinates are correlated by rho.
                    All the other co-ordinates are uncorrelated
    """
    z = np.random.normal(size = (n, d))
    sigma = np.eye(2)
    sigma[0, 1] = rho
    sigma[1, 0] = rho
    sigma_sqrt = sqrtm(sigma)
    z[:, :2] = z[:, :2] @ sigma_sqrt
    return z

def h(z, R, rho1 = 0):
    """
    Entangled representations
    Parameters
    ----------
    rho1 : float
        Lower triangular entries for lower triangular matrix
    """
    d = z.shape[1]
    L = np.eye(d)
    for i in range(d):
        for j in range(d):
            if i > j:
                L[i, j] = rho1
                
    return z @ R.T @ L.T

def latent_modify(z, i = 0, rho2 = 0):
    """
    Create modified latents
    
    returns positive + negative samples in terms of i-th style factors
    """
    n = z.shape[0]
    z1 = np.copy(z)
    z2 = np.copy(z)


    if i == 0:
        delta1 = np.abs(z[:, i]) - (z[:, i])
        delta2 = - np.abs(z[:, i]) - (z[:, i])
        z1[:, 0] = z1[:, 0] + delta1
        z1[:, 1] = z1[:, 1] + rho2 * delta1
        z2[:, 0] = z2[:, 0] + delta2
        z2[:, 1] = z2[:, 1] + rho2 * delta2

    elif i == 1:
        delta1 = np.abs(z[:, i]) - (z[:, i])
        delta2 = - np.abs(z[:, i]) - (z[:, i])
        z1[:, 1] = z1[:, 1] + delta1
        z1[:, 0] = z1[:, 0] + rho2 * delta1
        z2[:, 1] = z2[:, 1] + delta2
        z2[:, 0] = z2[:, 0] + rho2 * delta2
    else:
        z1[:, i] = np.abs(z[:, i])
        z2[:, i] = - np.abs(z[:, i])
        
    return z1, z2




def expt(n = 10000, d = 10, d_style = 5, rho1 = 0, rho2 = 0, lam = 0, seed = 0):
    """
    Runs a single instance of the experiment
    
    Parameters
    ----------
    n : int
        Sample size
    d : int
        Data dimension

    Returns
    -------
    ndarray
        Description of return value
    """
    # orthogonal matrix
    R = stats.ortho_group.rvs(d, random_state=0)
    
    # number of content factors
    d_content = d - d_style
    
    # generate data
    np.random.seed(seed)
    z = gen_latent(n, d, rho2)
    
    # disentangled directions
    alpha_style = np.zeros(shape = (d, d_style)) # style factor directions
    delta_list = []
    for i in range(d_style):
        # positive + negative samples
        z_pos, z_neg = latent_modify(z, i, rho2 = rho2)
        u_pos, u_neg = h(z_pos, R, rho1), h(z_neg, R, rho1)
        
        u_all = np.concatenate([u_pos, u_neg], axis = 0)
        y_all = np.array([1] * n + [-1] * n)
        
        # regression for style factor
        lr = LinearRegression().fit(u_all, y_all)
        coef = lr.coef_ 
        alpha_style[:, i] = coef / np.linalg.norm(coef)
        
        # differences for content factors
        delta_list.append(u_pos - u_neg)
        
        
    
    
    # PCA for content factors
    u = h(z, R, rho1)
    Sigma1 = u.T @ u / u.shape[0] # first part of PCA 

    
    Delta = np.concatenate(delta_list, axis = 0)
    Sigma2 = Delta.T @ Delta / Delta.shape[0] # second part of PCA 
    
    M = Sigma1 - lam * Sigma2 # PCA objective

    w, v = np.linalg.eigh(M) # eigen decomposition 
    alpha_content = v[:, (-d_content):]   # content factor directions    
    
    
    # sparse recovery evaluation on test data
    n_test = 40000
    z_test = gen_latent(n_test, d, rho2) # latent factors
    u_test = h(z_test, R, rho1)
    
    
    z_style = z_test[:, :d_style] # true style factors
    z_style_estimated = u_test @ alpha_style # estimated style factors
    z_content_estimated = u_test @ alpha_content # estimated content factors
    
    
    z_total = np.concatenate([z_style, z_style_estimated, z_content_estimated], axis = 1) # dimension concatenations
    corr = np.corrcoef(z_total.T) # full correlation matrix
    
    # true style correlation matrix
    style_corr = np.eye(d_style)
    style_corr[0, 1] = rho2
    style_corr[1, 0] = rho2
    
    style_recovery = np.linalg.norm(corr[:d_style, (d_style):(d_style * 2)] - style_corr) / d_style
    style_content_disentagle = np.linalg.norm(corr[:d_style, (2 * d_style):]) / np.sqrt(d_style * d_content)
    
    
    return style_recovery, style_content_disentagle


if __name__ == "__main__":
    ITERS = range(50)
    lams = [0, 0.01, 0.1, 1, 10, 100, 1000, 1e4]
    rho1s = [0.1, 0.5, 0.9]
    rho2s = [0.1, 0.5, 0.9]
    grid = list(product(rho1s, rho2s, lams, ITERS))
    
    i = int(float(sys.argv[1]))
        
    rho1, rho2, lam, ITER = grid[i]

    style_recovery, style_content_disentagle = expt(lam = lam,
                                                           seed = ITER,
                                                           d = 10, d_style = 5,
                                                           rho1 = rho1, rho2 = rho2)

    summary_dict = {'lam': lam, 'rho1': rho1, 'rho2': rho2, 'ITER': int(ITER),
                    'd': 10, 'style_recovery': float(style_recovery), 'style_content_disentagle': float(style_content_disentagle)}

    print(summary_dict)
    with open(f'summary/summary_{i}.json', 'w') as fp:
        json.dump(summary_dict, fp)

        
        
    
