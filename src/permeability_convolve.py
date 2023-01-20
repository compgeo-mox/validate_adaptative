import numpy as np
from scipy import signal
from scipy import interpolate

def compute_permeability(eps, u_bar, phi, Phi, ranges, a_val=2, num=1000, return_all=False):

    u_bar = np.atleast_1d(u_bar)

    if return_all:
        K_inv, a, G_prime, Psi = do_convolution(eps, u_bar, Phi, ranges, a_val, num, return_all)
    else:
        K_inv, a = do_convolution(eps, u_bar, Phi, ranges, a_val, num, return_all)

    # create an interpolator for the resulting inverse of permeability,
    # extending the values before and after
    I_K_inv = interpolate.interp1d(a/2, K_inv, kind="cubic")
    # create a lambda function to be actual permeability
    K = lambda a: K_fct(I_K_inv, phi[0], phi[1], u_bar[0], a, a_val)
    #K = lambda a: 1./I_K_inv(a)

    if return_all:
        return K, G_prime, Psi
    else:
        return K

def do_convolution(eps, u_bar, Phi, ranges, a_val, num, return_all):
    # define the two terms
    G_prime = lambda b: -b*np.exp(-0.5*np.square(b/eps))/(np.power(eps, 3)*np.sqrt(2*np.pi))
    #Psi = lambda b: Phi_1(b) * (b <= u_bar*u_bar) + Phi_2(b) * (b > u_bar*u_bar)
    Psi = lambda b: np.sum([P(b) * r(b) for P, r in zip(Phi, ranges)], axis=0)

    # define the space where to sample the two functions
    b = np.linspace(-a_val, a_val, num)
    b2 = np.linspace(-a_val, a_val, 2*num)

    # compute the convolution
    dx = np.abs(b[1] - b[0])
    if return_all:
        return signal.convolve(2*G_prime(b), Psi(b2), mode="same")*dx, b, G_prime, Psi
    else:
        return signal.convolve(2*G_prime(b), Psi(b2), mode="same")*dx, b

def K_fct(I_K_inv, phi_1, phi_2, u_bar, a, a_val):

    a = np.atleast_1d(a)
    # condition
    less = a + u_bar*u_bar <= -a_val
    more = a + u_bar*u_bar >= a_val
    between = np.logical_and(a + u_bar*u_bar > -a_val, a + u_bar*u_bar < a_val)

    K = np.zeros(a.size)
    K[less] = 1. / phi_1(a[less])
    K[more] = 1. / phi_2(a[more])
    K[between] = 1. / I_K_inv(a[between])

    return K
