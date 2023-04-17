import numpy as np
import porepy as pp
from functools import partial

import sys; sys.path.insert(0, "../../src/")
import tags
from permeability_convolve import *

class Data:

    def __init__(self, spe10, epsilon=None, u_bar=None, region=None, tol=1e-6):
        self.spe10 = spe10
        self.tol = tol

        # it is possible to have or epsilon and u_bar or the region file name
        if region is not None:
            self.region = np.loadtxt(region).astype(bool)
        else:
            self.region = None

        # posso mettere la k adapt e le altre due come casi particolari usando pero' gli stessi dati
        u_bar = 1e-7
        epsilon = 1e-2

        mu = 0.001 # fluid's viscosity [Pa.s]
        rho = 1000. # fluid's density [kg/m3]
        cf = 0.55 # Forchheimer coefficient [-]
        K = 8e3 # characteristic permeability [m2]

        # Darcy
        lambda_1 = mu
        beta_1 = 0 *u_bar
        phi_1 = lambda a: lambda_1 + beta_1*np.sqrt(np.abs(a))
        pphi_1 = lambda a: lambda_1*a + 2/3*beta_1*np.power(np.abs(a), 1.5) # primitive of phi_1
        Phi_1 = lambda a: pphi_1(a)
        range_1 = lambda a: np.logical_and(a >= 0, a <= 1)

        # Forsh
        lambda_2 = mu
        beta_2 = cf*rho*np.sqrt(K) *u_bar
        phi_2 = lambda a: lambda_2 + beta_2*np.sqrt(np.abs(a))
        pphi_2 = lambda a: lambda_2*a + 2/3*beta_2*np.power(np.abs(a), 1.5) # primitive of phi_2
        Phi_2 = lambda a: pphi_2(a) + pphi_1(1) - pphi_2(1)
        range_2 = lambda a: a > 1

        # get permeability
        phi = [phi_1, phi_2]
        Phi = [Phi_1, Phi_2]
        ranges = [range_1, range_2]
        self.k_ref = compute_permeability(epsilon, ranges, phi=phi, Phi=Phi)

        self.k_adapt = lambda flux2: self.k_ref(flux2) / u_bar
        self.k_darcy = lambda _: 1/lambda_1 / u_bar
        self.k_forsh = lambda flux2: 1/phi_2(flux2) / u_bar

        # well data
        self.well_radius = 5 * pp.CENTIMETER
        self.well_perm = 1000 # TO FIX THIS PARAMETER

    # ------------------------------------------------------------------------------#

    def get(self):
        return {"k": self.perm,
                "bc": self.bc,
                "source": self.source,
                "vector_source": self.vector_source,
                "well_radius": self.well_radius,
                "tol": self.tol}

    # ------------------------------------------------------------------------------#

    def perm(self, sd, d, flow_solver):
        # set a fake permeability for the 0d grids
        if sd.dim == 0:
            return np.zeros(sd.num_cells)

        if sd.dim == 1 and sd.well_num > -1:
            return np.pi * np.square(self.well_radius) * np.ones(sd.num_cells) * self.well_perm

        # cell flux
        flux = d[pp.STATE][flow_solver.P0_flux]
        flux_norm2 = np.square(np.linalg.norm(flux, axis=0))

        if self.region is not None:
            k = np.zeros(sd.num_cells)
            # region zero is forch, region one is darcy
            darcy_region = self.region
            forsh_region = np.logical_not(self.region)

            k[darcy_region] = self.k_darcy(flux_norm2[darcy_region])
            k[forsh_region] = self.k_forsh(flux_norm2[forsh_region])
        else:
            k = self.k_adapt(flux_norm2)

        return k * self.spe10.perm[:, 0] * pp.DARCY

    # ------------------------------------------------------------------------------#

    def source(self, sd, d, flow_solver):
        return np.zeros(sd.num_cells)

    # ------------------------------------------------------------------------------#

    def vector_source(self, sd, d, flow_solver):

        if sd.dim == 0:
            return np.zeros((sd.num_cells, 3))

        perm = self.perm(sd, d, flow_solver)
        coeff = 0 * sd.cell_volumes.copy() / perm

        vect = np.vstack(
                (coeff, np.zeros(sd.num_cells), np.zeros(sd.num_cells))
                ).ravel(order="F")
        return vect

    # ------------------------------------------------------------------------------#

    def bc(self, sd, data, tol, flow_solver):

        # define the labels and values for the boundary faces
        bc_val = np.zeros(sd.num_faces)

        if sd.dim == 1 and sd.well_num > -1:
            b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            labels = np.array(["dir"])
            if sd.well_num == 0:
                bc_val[b_faces] = 1e7
            else:
                bc_val[b_faces] = 5e7

        else:
            b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            labels = np.array(["neu"] * b_faces.size)

        return labels, bc_val

    # ------------------------------------------------------------------------------#