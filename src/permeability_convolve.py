import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

def compute_permeability(eps, ranges, phi=None, Phi=None, coeffs=None, return_all=False):

    # convolution parameters to choose depending on eps and phi
    a_val = 20 # largest square flux allowed in convolved permeability
    num = int(2*np.floor(1e6/2)+1) # convolution resolution, always odd to center convolution at 0

    # compute inverse permeability from convolution and interpolation
    if coeffs == None:
        ranges, phi, Phi = prepare_convolution(ranges, phi=phi, Phi=Phi)
        if return_all:
            I_K_inv, a, G_prime, Psi = do_convolution(eps, a_val, num, return_all, \
                                                      ranges=ranges, phi=phi, Phi=Phi)
        else:
            I_K_inv, a = do_convolution(eps, a_val, num, return_all, \
                                        ranges=ranges, phi=phi, Phi=Phi)           

        # create a lambda function to be actual permeability
        K = lambda a: K_fct(I_K_inv, a, a_val, False)
    else:
        coeffs, powers = prepare_convolution(ranges, coeffs=coeffs)
        if return_all:
            I_K_inv, a, G_prime = do_convolution(eps, a_val, num, return_all, \
                                                 coeffs=coeffs, powers=powers)
        else:
            I_K_inv, a = do_convolution(eps, a_val, num, return_all, coeffs=coeffs, powers=powers)
        
        # create a lambda function to be actual permeability
        K = lambda a: K_fct(I_K_inv, a, a_val, True)

    if return_all:
        return K, G_prime, Psi
    else:
        return K

def prepare_convolution(ranges, phi=None, Phi=None, coeffs=None):
    # add artificial negative-speed region (for convolution)
    range_0 = lambda b: b < 0
    ranges = [range_0] + ranges

    if coeffs == None:
        # add Phi for negative flux needed for the convolution
        Phi_0 = lambda b, Phi=Phi: phi[0](0)*b + Phi[0](0) # to preserve differentiability at 0
        Phi = [Phi_0] + Phi
        
        return ranges, phi, Phi
    
    else:
        highest_order = len(coeffs[0])
        num_regions = len(ranges)-1
        num_cells = len(coeffs[0][0])

        # add coefficients of order 0 (constants)
        coeffs_cst = np.zeros(num_cells)
        coeffs[0] = [coeffs_cst] + coeffs[0]
        for i in range(num_regions-1):
            coeffs_cst = 2*np.sum(1/(j+2) * (coeffs[i][j+1] - coeffs[i+1][j]) \
                                  for j in range(highest_order))
            coeffs_cst += coeffs[i][0]  
            coeffs[i+1] = [coeffs_cst] + coeffs[i+1]
            
        # add coefficients of negative-speed region (region 0)
        coeffs_0 = [np.zeros(num_cells) for j in range(highest_order+1)]
        coeffs_0[0] = coeffs[0][0]
        coeffs_0[1] = coeffs[0][1]
        coeffs = [coeffs_0] + coeffs
        
        # build powers for polynomial laws
        powers = []
        for i in range(num_regions+1):
            powers_temp = [lambda b, i=i, j=j: \
                           2/(j+2)*np.sign(b)*np.power(np.abs(b),(j+2)/2)*ranges[i](b) \
                           for j in range(highest_order)]
            powers_temp = [lambda b, i=i: 1*ranges[i](b)] + powers_temp
            powers = powers + [powers_temp]

        return coeffs, powers

def do_convolution(eps, a_val, num, return_all, \
                   ranges=None, phi=None, Phi=None, coeffs=None, powers=None):
    # define the derivative of Gaussian to convolve
    G_prime = lambda b: -b*np.exp(-0.5*np.square(b/eps))/(np.power(eps, 3)*np.sqrt(2*np.pi))

    # define the space where to sample the two functions
    b = np.linspace(-2*a_val, 2*a_val, num)
    b2 = np.linspace(-2*a_val, 2*a_val, 2*num)
        
    # parameters for convolution
    dx = np.abs(b[1]-b[0])
    idx0 = int(0.5*(num+1)-1) # index such that b[idx0] = 0
    
    if coeffs == None:
        # define the second term to convolve
        Psi = lambda b: np.sum([P(b) * r(b) for P, r in zip(Phi, ranges)], axis=0)
        
        # compute the convolution
        conv = signal.convolve(2*G_prime(b), Psi(b2), mode="same")*dx
        
        # interpolate the convolution to get function defined from b=0 to b=a_val 
        I_K_inv = interpolate.interp1d(b[idx0:]/2, conv[idx0:], kind="cubic")
        
        # check I_K_inv and Psi match on plot
        psi = lambda b: np.sum([p(b) * r(b) for p, r in zip(phi, ranges[1:])], axis=0)
        # I_K_inv = psi
        b_new = b[idx0:]/2
        fig1 = plt.figure(1)
        plt.xlim([0, 2])
        plt.ylim([psi(0)-1.e-6, psi(2)+1.e-6])
        plt.plot(b_new, I_K_inv(b_new))
        plt.plot(b_new, psi(b_new))
        plt.savefig('inv_perm.png')
        print('inverse permeability plot done')
        print('convolution Linfty error: ', \
              np.max(np.abs( (I_K_inv(b_new)-psi(b_new)) ))/np.max(np.abs( psi(b_new) )))
        print('convolution L1 error: ', \
              np.sum(np.abs( (I_K_inv(b_new)-psi(b_new)) ))/np.sum(np.abs( psi(b_new) )))
        
        if return_all:
            return I_K_inv, b, G_prime, Psi
        else:
            return I_K_inv, b

    else:
        highest_order = len(coeffs[0])-1
        num_regions = len(coeffs)-1
        num_cells = len(coeffs[0][0])

        # compute the convolution and interpolation simultaneously
        I_conv_reg = []
        for i in range(num_regions+1):
            conv_pow_i = [signal.convolve(2*G_prime(b), powers[i][j](b2), mode="same")*dx \
                          for j in range(highest_order+1)]
            I_pow_i = [interpolate.interp1d(b[idx0:]/2, conv_pow_i[j][idx0:], kind="cubic") \
                       for j in range(highest_order+1)]
            I_conv_i = []
            for k in range(num_cells):
                I_conv_i = I_conv_i + [add_fcts([multiply_fct(coeffs[i][j][k],I_pow_i[j]) \
                                                 for j in range(highest_order+1)])]
            I_conv_reg = I_conv_reg + [I_conv_i]

        I_K_inv = []
        for k in range(num_cells):
            I_K_inv = I_K_inv + [add_fcts([I_conv_reg[i][k] for i in range(num_regions+1)])]
    
        if return_all:
            return I_K_inv, b, G_prime
        else:
            return I_K_inv, b

def K_fct(I_K_inv, a, a_val, space_dependent_law):

    a = np.atleast_1d(a)
    K = np.zeros(a.size)
        
    if not space_dependent_law:
        # conditions
        less = a < 0 # will not happen since a = flux^2 >= 0
        more = a > a_val
        between = np.logical_and(a >= 0, a <= a_val)
    
        # K
        K[less] = 1. / I_K_inv(0)
        K[more] = 1. / I_K_inv(a_val)
        K[between] = 1. / I_K_inv(a[between])

    else:
        # K
        for k in range(a.size):
            if a[k] < 0:
                K[k] = 1. / I_K_inv[k](0)
            elif a[k] > a_val:
                K[k] = 1. / I_K_inv[k](a_val)
            else:
                K[k] = 1. / I_K_inv[k](a[k])

    return K

def multiply_fct(scalar, fct):
    return lambda x: scalar*fct(x)

def add_fcts(fcts):
    return lambda x: np.sum(fcts[i](x) for i in range(len(fcts)))
