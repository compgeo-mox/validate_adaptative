import numpy as np
import porepy as pp

import sys

sys.path.insert(0, "./src/")
# import tags
from perm_factor import *
from weighted_norm import *

# ------------------------------------------------------------------------------#


class Data:
    def __init__(self, parameters, problem, folder, tol=1e-6):
        self.problem = problem
        self.parameters = parameters
        self.tol = tol
        self.folder = folder

        # get necessary parameters
        u_bar = self.problem.u_bar
        m = self.parameters.m
        nu = self.parameters.nu
        c_F = self.parameters.c_F
        diss = self.parameters.dissipative
        if diss:
            factor = nu/(np.power(c_F, 2/(m - 1)))
            alpha = factor * u_bar 
            beta = factor * np.power(u_bar, m)
        else:
            alpha = nu * u_bar  # multiply by u_bar for normalization
            beta = nu * c_F * np.power(u_bar, m)  # idem
        zero = 0.0  # to be put as second-order term in Darcy region

        # convert drag coefficients to arrays if intrinsic permeability is heterogeneous
        mu = self.parameters.mu
        kappa = problem.kappa
        homogeneous_perm = True if np.unique(kappa).size == 1 else False

        if (homogeneous_perm 
            and (np.asarray(alpha).size == 1 and np.asarray(beta).size == 1)):
            if diss:
                denom = np.square(kappa[0]/mu)
                alpha /= denom
                beta /= denom
            else:
                beta *= np.power(kappa[0]/mu, m-1)
        else:
            if diss:
                denom = np.square(kappa/mu)
                alpha /= denom
                beta /= denom
            else:
                alpha *= np.ones(kappa.size)
                beta *= np.power(kappa/mu, m-1)
            zero *= np.ones(kappa.size)

        # gather all law coefficients in one list
        M = int(np.floor(m))
        self.coeffs = [[alpha] + [zero for j in range(1, M)], 
                       [alpha] + [zero for j in range(1, M - 1)] + [beta]]

        # ranges to define regions (normalized by u_bar)
        range_1 = lambda a: np.logical_and(a >= 0, a <= 1)  # slow-flux region (Darcy)
        range_2 = lambda a: a > 1  # high-flux region (Forchheimer)
        self.ranges = [range_1, range_2]

    # ------------------------------------------------------------------------------#

    def get_perm_factor(self, region=None):
        # compute flux-dependent perm factor by convolution (region is None) or region-wise
        m = self.parameters.m
        if region is None:  # use convolution
            self.region = None

            # adaptive convolution scheme
            k_adapt = perm_factor(self.coeffs, m, ranges=self.ranges)
            self.k_adapt = lambda flux2: k_adapt(flux2)
        else:  # use region file name
            self.region = np.loadtxt(self.folder + "regions/" + region).astype(bool)

            # region zero is Forchheimer, region one is Darcy
            darcy_region = self.region
            forch_region = np.logical_not(self.region)
            k_darcy = perm_factor(self.coeffs[0], m, region=darcy_region)
            k_forch = perm_factor(self.coeffs[1], m, region=forch_region)
            self.k_darcy = lambda flux2: k_darcy(flux2)
            self.k_forch = lambda flux2: k_forch(flux2)

    # ------------------------------------------------------------------------------#

    def effective_perm(self, sd, d, flow_solver):
        perm = self.problem.perm
        perm_diss = self.problem.perm_diss
        diss = self.parameters.dissipative

        # set a fake permeability for the 0d grids
        if sd.dim == 0:
            return np.zeros(sd.num_cells)

        # cell flux
        flux = d[pp.TIME_STEP_SOLUTIONS][flow_solver.P0_flux][0]
        if diss:
            flux_norm2 = np.square(weighted_norm(flux, perm_diss, sd.dim))
        else:
            flux_norm2 = np.square(np.linalg.norm(flux, axis=0))

        # retrieve flux-dependent perm factor and multiply it by intrinsic permeability
        if self.region is None:
            k = self.k_adapt(flux_norm2)
        else:
            k = np.zeros(sd.num_cells)
            # region zero is Forchheimer, region one is Darcy
            darcy_region = self.region
            forch_region = np.logical_not(self.region)
            k[darcy_region] = self.k_darcy(flux_norm2[darcy_region])
            k[forch_region] = self.k_forch(flux_norm2[forch_region])

        # return k multiplied by diagonal intrinsic permeability
        if diss:
            k_mult = [np.multiply(k, perm_diss[:, 0])]
            for j in range(1, sd.dim):
                k_mult.append(np.multiply(k, perm_diss[:, j]))
        else:
            k_mult = [np.multiply(k, perm[:, 0])]
            for j in range(1, sd.dim):
                k_mult.append(np.multiply(k, perm[:, j]))
        return k_mult

    # ------------------------------------------------------------------------------#

    def get(self):
        return {
            "k": self.effective_perm,
            "bc": self.bc,
            "source": self.source,
            "vector_source": self.vector_source,
            "tol": self.tol,
            "perm_diss": self.problem.perm_diss,
            "dissipative": self.parameters.dissipative
        }

    # ------------------------------------------------------------------------------#

    def source(self, sd, d, flow_solver):
        wells = self.parameters.wells
        u_bar = self.problem.u_bar

        vals = np.zeros(sd.num_cells)
        for well in wells:
            well_cell = well["cell_id"]
            vals[well_cell] = well["val"]

        return vals / u_bar

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
        bc = self.parameters.bdry_conditions
        labels = np.array([bc] * b_faces.size)
        bc_val = np.zeros(sd.num_faces)

        if bc == "dir":
            bc_val[b_faces] = atm_pressure + rho * g * h  # hydrostatic pressure all around

        return labels, bc_val

    # ------------------------------------------------------------------------------#
