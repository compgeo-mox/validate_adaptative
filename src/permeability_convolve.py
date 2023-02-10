import numpy as np
from scipy import signal
from scipy import interpolate

def compute_permeability(eps, phi, Phi, ranges, return_all=False):

    # convolution parameters to choose depending on eps and phi
    a_val = 10 # largest square flux allowed in convolved permeability
    num = int(2*np.floor(1e5/2)+1) # convolution resolution, always odd to center convolution at 0

    # compute inverse permeability from convolution and interpolation
    if return_all:
        I_K_inv, a, G_prime, Psi = do_convolution(eps, Phi, ranges, a_val, num, return_all)
    else:
        I_K_inv, a = do_convolution(eps, phi, Phi, ranges, a_val, num, return_all)
        
    # create a lambda function to be actual permeability
    K = lambda a: K_fct(I_K_inv, a, a_val)

    if return_all:
        return K, G_prime, Psi
    else:
        return K

def do_convolution(eps, phi, Phi, ranges, a_val, num, return_all):
    # add Phi and range for negative flux needed for the convolution
    Phi_0 = lambda b: phi[0](0)*b + Phi[0](0) # to preserve differentiability at 0
    range_0 = lambda b: b < 0
    Phi_all = [Phi_0] + Phi
    ranges_all = [range_0] + ranges
    
    # define the two terms to convolve
    G_prime = lambda b: -b*np.exp(-0.5*np.square(b/eps))/(np.power(eps, 3)*np.sqrt(2*np.pi))
    Psi = lambda b: np.sum([P(b) * r(b) for P, r in zip(Phi_all, ranges_all)], axis=0)

    # define the space where to sample the two functions
    b = np.linspace(-2*a_val, 2*a_val, num)
    b2 = np.linspace(-2*a_val, 2*a_val, 2*num)

    # compute the convolution
    dx = np.abs(b[1]-b[0])
    conv = signal.convolve(2*G_prime(b), Psi(b2), mode="same")*dx

    # interpolate the convolution to get function defined from b=0 to b=a_val 
    idx0 = int(0.5*(num+1)-1) # index such that b[idx0] = 0
    I_K_inv = interpolate.interp1d(b[idx0:]/2, conv[idx0:], kind="cubic")

    # check I_K_inv and Psi match on plot
    psi = lambda b: np.sum([p(b) * r(b) for p, r in zip(phi, ranges)], axis=0)
    #I_K_inv = psi
    b_new = b[idx0:]/2
    import matplotlib.pyplot as plt
    plt.xlim([0, 2])
    plt.plot(b_new, I_K_inv(b_new))
    plt.plot(b_new, psi(b_new))
    plt.savefig('inv_perm.png')
    print('inverse permeability plot done')
    print('convolution Linfty error: ', np.max(np.abs( (I_K_inv(b_new)-psi(b_new)) ))/np.max(np.abs( psi(b_new) )))
    print('convolution L1 error: ', np.sum(np.abs( (I_K_inv(b_new)-psi(b_new)) ))/np.sum(np.abs( psi(b_new) )))
    
    if return_all:
        return I_K_inv, b, G_prime, Psi
    else:
        return I_K_inv, b

def K_fct(I_K_inv, a, a_val):

    a = np.atleast_1d(a)

    # conditions
    less = a < 0 # will not happen since a = flux^2 >= 0
    more = a > a_val
    between = np.logical_and(a >= 0, a <= a_val)

    # K
    K = np.zeros(a.size)
    K[less] = 1. / I_K_inv(0)
    K[more] = 1. / I_K_inv(a_val)
    K[between] = 1. / I_K_inv(a[between])

    return K
