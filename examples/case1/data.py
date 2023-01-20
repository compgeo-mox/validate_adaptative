import numpy as np
import porepy as pp
from functools import partial

import sys; sys.path.insert(0, "../../src/")
import tags
from permeability_convolve import *

class Data:

    def __init__(self, epsilon, u_bar, spe10, tol=1e-6):
        self.epsilon = epsilon
        self.u_bar = u_bar
        self.spe10 = spe10
        self.tol = tol

        num = 100000

        # Darcy: data 1/k_1 and 1/k_2
        lambda_1 = 1
        beta_1 = 0
        phi_1 = lambda a: lambda_1 + beta_1*np.sqrt(np.abs(a))
        Phi_1 = lambda a: lambda_1*(a-u_bar*u_bar) + beta_1*(np.power(np.abs(a), 1.5) - np.power(u_bar, 3))*2/3
        range_1 = lambda a: a <= u_bar*u_bar

        # Forsh
        lambda_2 = 0.1
        beta_2 = 100 # 10 100 500 1000
        phi_2 = lambda a: lambda_2 + beta_2*np.sqrt(np.abs(a))
        Phi_2 = lambda a: lambda_2*(a-u_bar*u_bar) + beta_2*(np.power(np.abs(a), 1.5) - np.power(u_bar, 3))*2/3
        range_2 = lambda a: a > u_bar*u_bar


        phi = [phi_1, phi_2]
        Phi = [Phi_1, Phi_2]
        ranges = [range_1, range_2]
        self.k_ref = compute_permeability(self.epsilon, self.u_bar/self.u_bar, phi, Phi, ranges, num=num)

        #import matplotlib.pyplot as plt
        #u = np.linspace(0, 2, num)
        #plt.plot(u, self.k_ref(u))
        #plt.show()

        self.k = lambda flux2: self.k_ref(flux2/self.u_bar)

    # ------------------------------------------------------------------------------#

    def get(self):
        return {"k": self.perm,
                "bc": self.bc,
                "source": self.source,
                "vector_source": self.vector_source,
                "tol": self.tol}

    # ------------------------------------------------------------------------------#

    def perm(self, g, d, flow_solver):
        # set a fake permeability for the 0d grids
        if g.dim == 0:
            return np.zeros(g.num_cells)

        # cell flux
        flux = d[pp.STATE][flow_solver.P0_flux]
        flux_norm2 = np.square(np.linalg.norm(flux, axis=0))

        return self.k(flux_norm2) * self.spe10.perm[:, 0] * pp.DARCY

    # ------------------------------------------------------------------------------#

    def source(self, g, d, flow_solver):
        return np.zeros(g.num_cells)

    # ------------------------------------------------------------------------------#

    def vector_source(self, g, d, flow_solver):

        if g.dim == 0:
            return np.zeros((g.num_cells, 3))

        perm = self.perm(g, d, flow_solver)
        coeff = 0*5e-2 * g.cell_volumes.copy() / perm

        vect = np.vstack(
                (coeff, np.zeros(g.num_cells), np.zeros(g.num_cells))
                ).ravel(order="F")
        return vect

    # ------------------------------------------------------------------------------#

    def bc(self, g, data, tol, flow_solver):
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        b_face_centers = g.face_centers[:, b_faces]

        # define outflow type boundary conditions
        out_flow = b_face_centers[1] > self.spe10.full_physdims[1] - tol

        # define inflow type boundary conditions
        in_flow = b_face_centers[1] < 0 + tol

        # define the labels and values for the boundary faces
        labels = np.array(["neu"] * b_faces.size)
        bc_val = np.zeros(g.num_faces)

        labels[in_flow] = "dir"
        labels[out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1e7

        return labels, bc_val

    # ------------------------------------------------------------------------------#
