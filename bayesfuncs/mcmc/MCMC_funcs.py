import numpy as np
from time import time
from autograd import grad, hessian

def MCMC(niter, loglik, logprior, x0, label='Whittle', acceptance_lag=1000, **loglikargs):

    def logpost(x): return  loglik(x, **loglikargs) + logprior(x)
    def obj(x):     return -logpost(x)
    
    from scipy.optimize import minimize
    MAP = minimize(obj, jac = grad(obj), method = 'L-BFGS-B', x0 = x0)
    if not MAP["success"]:
        warnings.warn("Optimizer didn't converge")
        
    print(f'{label} MAP:  {np.round(MAP.x,3)}')
    
    propcov = np.linalg.inv(-hessian(logpost)(MAP.x))
    # return MAP, propcov, 1   # return and set seed if you need to compare
    postdraws, A = RW_MH(niter, MAP.x, propcov, logpost, acceptance_lag=acceptance_lag)
    return MAP, propcov, postdraws

def RW_MH(niter, x0, propcov, posterior, acceptance_lag=1000):
    
    n_params  = len(x0)
    h         = 2.38/np.sqrt(n_params)
    
    A         = np.zeros(niter, dtype=np.float64)
    U         = np.random.rand(niter)
    props     = h*np.random.multivariate_normal(np.zeros(len(propcov)), propcov, size=niter)
    post      = np.zeros((niter, n_params))
    
    post[0]   = x0
    crnt_step = x0
    bottom    = posterior(crnt_step)
    # print(bottom)
    
    print(f'{"initializing RWMH":-^50}')
    t0 = time()
    for i in range(1, niter):
        
        prop_step = crnt_step + props[i]
        top       = posterior(prop_step)
        # print(top)
        
        A[i]      = np.min((1., np.exp(top-bottom)))
        if U[i] < A[i]:
            crnt_step  = prop_step
            bottom     = top
        
        post[i]   = crnt_step
            
        if (i+1)%acceptance_lag==0:
            print(f'Iteration: {i+1}    Acceptance rate: {A[i-(acceptance_lag-1): (i+1)].mean().round(3)}    Time: {np.round(time()-t0,3)}s')
            
    return post, A
