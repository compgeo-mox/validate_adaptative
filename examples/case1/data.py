import numpy as np
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
import tags
from perm_factor import *

# ------------------------------------------------------------------------------#

class Data:

    def __init__(self, parameters, problem, tol=1e-6):
        self.problem = problem
        self.parameters = parameters
        self.tol = tol

        # get necessary parameters
        u_bar = self.problem.u_bar
        mu = self.parameters.mu *u_bar # multiply by u_bar for normalization
        beta = self.parameters.beta *np.square(u_bar) # idem
        zero = 0. # to be put as second-order term in Darcy region

        # convert drag coefficients to arrays if intrinsic permeability is heterogeneous
        bg_K = problem.perm[:, 0] # intrinsic permeability # [np.max(problem.perm[:, 0])]
        homogeneous_perm = True if np.unique(bg_K).size == 1 else False

        if homogeneous_perm is True and (np.asarray(mu).size == 1 and np.asarray(beta).size == 1):
            beta *= np.sqrt(bg_K[0])
        else:
            mu *= np.ones(bg_K.size)
            beta *= np.sqrt(bg_K)
            zero *= np.ones(bg_K.size)

        # gather all law coefficients in one list
        self.coeffs = [ [mu, zero], [mu, beta] ]

        # ranges to define regions (normalized by u_bar)
        range_1 = lambda a: np.logical_and(a >= 0, a <= 1) # slow-speed region (Darcy)
        range_2 = lambda a: a > 1 # high-speed region (Forchheimer)
        self.ranges = [range_1, range_2]

    # ------------------------------------------------------------------------------#

    def get_perm_factor(self, region=None):

        # compute speed-dependent perm factor by convolution (region is None) or region-wise
        if region is None: # use convolution
            self.region = None

            # adaptive convolution scheme
            k_adapt = perm_factor(self.coeffs, ranges=self.ranges)
            self.k_adapt = lambda flux2: k_adapt(flux2)
        else: # use region file name
            self.region = np.loadtxt("./regions/" + region).astype(bool)

            # region zero is Forchheimer, region one is Darcy
            darcy_region = self.region
            forch_region = np.logical_not(self.region)
            k_darcy = perm_factor(self.coeffs[0], region=darcy_region)
            k_forch = perm_factor(self.coeffs[1], region=forch_region)
            self.k_darcy = lambda flux2: k_darcy(flux2)
            self.k_forch = lambda flux2: k_forch(flux2)

    # ------------------------------------------------------------------------------#

    def effective_perm(self, sd, d, flow_solver):
        # set a fake permeability for the 0d grids
        if sd.dim == 0:
            return np.zeros(sd.num_cells)

        # cell flux
        flux = d[pp.STATE][flow_solver.P0_flux]
        flux_norm2 = np.square(np.linalg.norm(flux, axis=0))

        # retrieve speed-dependent perm factor and multiply it by intrinsic permeability
        if self.region is None:
            k = self.k_adapt(flux_norm2)
        else:
            k = np.zeros(sd.num_cells)
            # region zero is Forchheimer, region one is Darcy
            darcy_region = self.region
            forch_region = np.logical_not(self.region)
            k[darcy_region] = self.k_darcy(flux_norm2[darcy_region])
            k[forch_region] = self.k_forch(flux_norm2[forch_region])

        return k * self.problem.perm[:, 0] # multiply by intrinsic permeability

    # ------------------------------------------------------------------------------#

    def get(self):
        return {"k": self.effective_perm,
                "bc": self.bc,
                "source": self.source,
                "vector_source": self.vector_source,
                "tol": self.tol}

    # ------------------------------------------------------------------------------#

    def source(self, sd, d, flow_solver):
        wells = self.parameters.wells

        vals = np.zeros(sd.num_cells)
        for well in wells:
            well_cell = well["cell_id"]
            vals[well_cell] = well["val"]

        return vals

    # ------------------------------------------------------------------------------#

    def vector_source(self, sd, d, flow_solver):
        if sd.dim == 0:
            return np.zeros((sd.num_cells, 3))
        else:
            # no gravity
            vals = np.zeros((3, sd.num_cells))

            return vals.ravel(order="F")

    # ------------------------------------------------------------------------------#

    def bc(self, sd, data, tol, flow_solver):
        # get boundary faces
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]

        # get necessary parameters
        rho = self.parameters.rho
        atm_pressure = self.parameters.atm_pressure
        g = self.parameters.g
        h = self.parameters.layer_depth

        # define the labels and values for the boundary faces
        labels = np.array(["dir"] * b_faces.size)
        bc_val = np.zeros(sd.num_faces)

        bc_val[b_faces] = atm_pressure + rho*g*h # hydrostatic pressure all around

        return labels, bc_val

    # ------------------------------------------------------------------------------#
