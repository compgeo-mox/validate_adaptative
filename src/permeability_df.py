import numpy as np

def compute_permeability_df(coeffs, darcy_region, forch_region):
    lambda_1 = coeffs[0][0][darcy_region]
    lambda_2 = coeffs[1][0][forch_region]
    beta_2 = coeffs[1][1][forch_region]

    phi_darcy = [lambda a: lambda_1[i] for i in range(len(lambda_1))]
    phi_forch = [lambda a: lambda_2[i] + beta_2[i]*np.sqrt(np.abs(a)) \
                 for i in range(len(lambda_2))]

    K_darcy = lambda a: K_fct_darcy(phi_darcy, a)
    K_forch = lambda a: K_fct_forch(phi_forch, a)

    return K_darcy, K_forch
    
def K_fct_darcy(phi_darcy, a):
    a = np.atleast_1d(a)

    K_darcy = np.zeros(a.size)
        
    for k in range(a.size):
        K_darcy[k] = 1. / phi_darcy[k](a[k])
    
    return K_darcy

def K_fct_forch(phi_forch, a):
    a = np.atleast_1d(a)

    K_forch = np.zeros(a.size)
        
    for k in range(a.size):
        K_forch[k] = 1. / phi_forch[k](a[k])

    return K_forch
