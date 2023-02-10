import numpy as np
import porepy as pp
from functools import partial

import sys; sys.path.insert(0, "../../src/")
import tags
from permeability_convolve import *

class Data:

    def __init__(self, problem, epsilon=None, u_bar=None, region=None, tol=1e-6, num=100000):
        self.problem = problem
        self.tol = tol

        # it is possible to have or epsilon and u_bar or the region file name
        if region is not None:
            self.region = np.loadtxt(region).astype(bool)
        else:
            self.region = None

        # Darcy: data 1/k_1 and 1/k_2
        lambda_1 = 1
        beta_1 = 0
        phi_1 = lambda a: lambda_1 + beta_1*np.sqrt(np.abs(a))
        Phi_1 = lambda a: lambda_1*(a-u_bar*u_bar) + beta_1*(np.power(np.abs(a), 1.5) - np.power(u_bar, 3))*2/3
        range_1 = lambda a: a <= u_bar*u_bar

        # Forsh
        lambda_2 = 1
        beta_2 = 1 # 10 100 500 1000
        phi_2 = lambda a: lambda_2 + beta_2*np.sqrt(np.abs(a))
        Phi_2 = lambda a: lambda_2*(a-u_bar*u_bar) + beta_2*(np.power(np.abs(a), 1.5) - np.power(u_bar, 3))*2/3
        range_2 = lambda a: a > u_bar*u_bar

        phi = [phi_1, phi_2]
        Phi = [Phi_1, Phi_2]
        ranges = [range_1, range_2]
        self.k_ref = compute_permeability(epsilon, u_bar, phi, Phi, ranges, num=num)

        self.k_adapt = lambda flux2: self.k_ref(flux2)
        self.k_darcy = lambda _: 1/lambda_1
        self.k_forsh = lambda flux2: phi_2(flux2)

    # ------------------------------------------------------------------------------#

    def get(self):
        return {"k": self.perm,
                "bc": self.bc,
                "source": self.source,
                "vector_source": self.vector_source,
                "tol": self.tol}

    # ------------------------------------------------------------------------------#

    def perm(self, sd, d, flow_solver):
        # set a fake permeability for the 0d grids
        if sd.dim == 0:
            return np.zeros(sd.num_cells)

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

        return k * self.problem.perm[:, 0] #* pp.DARCY

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
        b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
        b_face_centers = sd.face_centers[:, b_faces]

        # define outflow type boundary conditions
        out_flow = b_face_centers[0] > self.problem.full_physdims[1] - tol

        # define inflow type boundary conditions
        in_flow = b_face_centers[0] < 0 + tol

        # define the labels and values for the boundary faces
        labels = np.array(["neu"] * b_faces.size)
        bc_val = np.zeros(sd.num_faces)

        labels[in_flow] = "dir"
        labels[out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1

        return labels, bc_val

    # ------------------------------------------------------------------------------#
