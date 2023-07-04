import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate

# ------------------------------------------------------------------------------#

def perm_factor(coeffs, ranges=None, region=None):
    assert not(ranges is None and region is None), "At least ranges or a region must be provided"

    # if no ranges are provided, then no convolution is done
    if ranges is None:
        # get number of cells, with convention 0 if law is not space-dependent
        num_cells = np.asarray(coeffs[0]).ndim * np.asarray(coeffs[0]).size
        law_order = len(coeffs) # highest order of law (darcy is 1 and forch is 2)

        # compute and return flux-dependent permeability factor
        K_inv = inv_perm(coeffs, region, num_cells, law_order)
        return lambda a: K_fct(a, K_inv, num_cells)

    else: # with convolution
        num_cells = np.asarray(coeffs[0][0]).ndim * np.asarray(coeffs[0][0]).size
        law_order = len(coeffs[0])
        num_regions = len(ranges)

        # compute and return permeability factor from convolution and interpolation
        coeffs_all, powers = convolution_terms(coeffs, ranges, num_cells, law_order, num_regions)
        I_K_inv, a_max = convolution(coeffs_all, ranges, powers, \
                                     num_cells, law_order, num_regions)
        return lambda a: K_fct(a, I_K_inv, num_cells, a_max=a_max)

# ------------------------------------------------------------------------------#

def inv_perm(coeffs, region, num_cells, law_order):
    # build (primitives of) powers for polynomial laws
    ppowers = [lambda a, j=j: np.power(np.abs(a),j/2) for j in range(law_order)]

    # compute inverse permeability factor by multiplying powers by coefficients and adding up
    if num_cells == 0: # law is not space_dependent
        K_inv = f_sum([f_mult(coeffs[j],ppowers[j]) for j in range(law_order)])
    else: # law is space-dependent
        coeffs_region = [coeffs[j][region] for j in range(law_order)]
        num_cells_region = len(coeffs_region[0])
        K_inv = [f_sum([f_mult(coeffs_region[j][k],ppowers[j]) \
                           for j in range(law_order)]) for k in range(num_cells_region)]

    return K_inv

# ------------------------------------------------------------------------------#

def convolution_terms(coeffs, ranges, num_cells, law_order, num_regions):
    # copy ranges and coeffs for safety
    ranges_all = ranges.copy()
    coeffs_all = coeffs.copy()

    # add artificial negative-flux region (for convolution)
    range_0 = lambda a: a < 0
    ranges_all = [range_0] + ranges

    # to initilize to zero depending on space-dependency of law
    init_zero = np.zeros(num_cells) if num_cells > 0 else 0

    # add coefficients of order 0 (constants)
    coeffs_cst = init_zero
    coeffs_all[0] = [coeffs_cst] + coeffs_all[0]
    for i in range(num_regions-1):
        coeffs_cst = 2*np.sum(1/(j+2)*(coeffs_all[i][j+1] - coeffs_all[i+1][j]) \
                              for j in range(law_order))
        coeffs_cst += coeffs_all[i][0]
        coeffs_all[i+1] = [coeffs_cst] + coeffs_all[i+1]

    # add coefficients of negative-flux region (region 0)
    coeffs_0 = [init_zero for j in range(law_order+1)]
    coeffs_0[0] = coeffs_all[0][0]
    coeffs_0[1] = coeffs_all[0][1]
    coeffs_all = [coeffs_0] + coeffs_all

    # build powers for polynomial laws
    powers = []
    for i in range(num_regions+1):
        powers_i = [lambda a, i=i, j=j: 2/(j+2)*a*np.power(np.abs(a),j/2)*ranges_all[i](a) \
                    for j in range(law_order)]
        powers_i = [lambda a, i=i: 1*ranges_all[i](a)] + powers_i
        powers = powers + [powers_i]

    return coeffs_all, powers

# ------------------------------------------------------------------------------#

def convolution(coeffs, ranges, powers, num_cells, law_order, num_regions):
    # convolution PARAMETERS #
    a_max = 20 # largest square flux allowed in convolved permeability factor
    n_conv = int(2*np.floor(1e6/2)+1) # convolution resolution, odd to center convolution at 0
    epsilon = 0.1 # refinement of convolution around jump
    # define the derivative of Gaussian to convolve
    G_prime = lambda b: \
        -b*np.exp(-0.5*np.square(b/epsilon))/(np.power(epsilon, 3)*np.sqrt(2*np.pi))
    ##

    # define the space where to sample the two functions for convolution
    b = np.linspace(-2*a_max, 2*a_max, n_conv)
    b2 = np.linspace(-2*a_max, 2*a_max, 2*n_conv)

    # extra parameters
    dx = np.abs(b[1]-b[0]) # step for convolution
    idx0 = int(0.5*(n_conv+1)-1) # index such that b[idx0] = 0
    b_pos = b[idx0:]/2 # interpolate, plot and get errors only over positive fluxes

    if num_cells == 0: # law is not space_dependent
        # define term to convolve (primitive of law), Phi
        Phi_region = []
        for i in range(num_regions+1):
            Phi_region = Phi_region + [f_sum([f_mult(coeffs[i][j],powers[i][j]) \
                                              for j in range(law_order+1)])]
            Phi = f_sum(Phi_region)

        # compute the convolution
        conv = signal.convolve(2*G_prime(b), Phi(b2), mode="same")*dx

        # interpolate the convolution to get function defined from b=0 to b=a_max
        I_K_inv = interpolate.interp1d(b_pos, conv[idx0:], kind="cubic")

    else: # law is space_dependent
        # compute the convolution and interpolation simultaneously
        I_conv_region = []
        for i in range(num_regions+1):
            conv_pow_i = [signal.convolve(2*G_prime(b), powers[i][j](b2), mode="same")*dx \
                          for j in range(law_order+1)]
            I_pow_i = [interpolate.interp1d(b_pos, conv_pow_i[j][idx0:], kind="cubic") \
                       for j in range(law_order+1)]
            I_conv_region = I_conv_region + [ [f_sum([f_mult(coeffs[i][j][k],I_pow_i[j]) \
                                                      for j in range(law_order+1)]) \
                                               for k in range(num_cells)] ]

        # get inverse permeability factor
        I_K_inv = [f_sum([I_conv_region[i][k] for i in range(num_regions+1)]) \
                   for k in range(num_cells)]

    # check I_K_inv and law match by plotting and getting error in convolution
    plot_and_get_errors(I_K_inv, coeffs, ranges, num_cells, law_order, num_regions, b_pos)

    return I_K_inv, a_max

# ------------------------------------------------------------------------------#

def K_fct(a, K_inv, num_cells, a_max=None):
    # a will be of size either the total number of cells or the number of cells in current region
    a = np.atleast_1d(a)
    K = np.zeros(a.size)

    if a_max is None: # no convolution was computed
        if num_cells == 0: # law is not space-dependent
            K = 1. / K_inv(a)
        else: # law is space-dependent
            for k in range(a.size):
                K[k] = 1. / K_inv[k](a[k])

    else: # convolution was computed (so tail of permeability factor needs to be approximated)
        if num_cells == 0: # law is not space-dependent
            less = a < 0 # will not happen since a = flux^2 >= 0
            more = a > a_max # for tail approximation
            between = np.logical_and(a >= 0, a <= a_max)

            # get K
            K[less] = 1. / K_inv(0)
            K[more] = 1. / K_inv(a_max)
            K[between] = 1. / K_inv(a[between])

        else: # law is space-dependent
            for k in range(a.size):
                if a[k] < 0:
                    K[k] = 1. / K_inv[k](0)
                elif a[k] > a_max:
                    K[k] = 1. / K_inv[k](a_max)
                else:
                    K[k] = 1. / K_inv[k](a[k])

    return K

# ------------------------------------------------------------------------------#

def plot_and_get_errors(I_K_inv, coeffs, ranges, num_cells, law_order, num_regions, b_pos):
    # build (primitives of) powers for polynomial laws
    ppowers = []
    for i in range(num_regions):
        ppowers_i = [lambda a, i=i, j=j: np.power(np.abs(a),j/2)*ranges[i](a) \
                    for j in range(law_order)]
        ppowers = ppowers + [ppowers_i]

    # define exact (non-convolved) inverse permeability factor, phi
    phi_region = []
    if num_cells == 0: # law is not space-dependent
        for i in range(num_regions):
            phi_region = phi_region + [f_sum([f_mult(coeffs[i+1][j+1],ppowers[i][j]) \
                                              for j in range(law_order)])]
            phi = f_sum(phi_region)
    else: # law is space-dependent
        for i in range(num_regions):
            phi_region = phi_region + [ [f_sum([f_mult(coeffs[i+1][j+1][k],ppowers[i][j]) \
                                                for j in range(law_order)]) \
                                         for k in range(num_cells)] ]
        phi = [f_sum([phi_region[i][k] for i in range(num_regions)]) \
               for k in range(num_cells)]

    # plot and get error in first cell
    phi = phi[0] if num_cells > 0 else phi
    I_K_inv = I_K_inv[0] if num_cells > 0 else I_K_inv

    # plot and get error
    fig1 = plt.figure(1)
    x_max = 2 # maximum on x-axis
    plt.xlim([0, x_max])
    plt.ylim([phi(0)-1.e-10, phi(x_max)+1.e-10])
    plt.plot(b_pos, I_K_inv(b_pos))
    plt.plot(b_pos, phi(b_pos))
    plt.savefig("./figures/inv_perm_fact_conv.png")
    print("plot of inverse permeability factor done")
    print("convolution Linfty error: ", \
          np.max(np.abs( (I_K_inv(b_pos)-phi(b_pos)) ))/np.max(np.abs( phi(b_pos) )))
    print("            L1 error: ", \
          np.sum(np.abs( (I_K_inv(b_pos)-phi(b_pos)) ))/np.sum(np.abs( phi(b_pos) )))

# ------------------------------------------------------------------------------#

def f_mult(scalar, fct):
    return lambda x: scalar*fct(x) # multiply a lambda fct by a scalar

# ------------------------------------------------------------------------------#

def f_sum(fcts):
    return lambda x: np.sum(fcts[i](x) for i in range(len(fcts))) # add up a list of lambda fcts

# ------------------------------------------------------------------------------#
