import numpy as np
import porepy as pp
from functools import partial

import sys; sys.path.insert(0, "../../src/")
import tags
from permeability_convolve import *

class Data:

    def __init__(self, problem, epsilon=None, region=None, tol=1e-6):
        self.problem = problem
        self.tol = tol

        # it is possible to have or epsilon and u_bar or the region file name
        if region is not None:
            self.region = np.loadtxt(region).astype(bool)
        else:
            self.region = None

        # physical parameters
        mu = 3e-4 # fluid's viscosity [Pa.s]
        rho = 970. # fluid's density [kg/m3]
        cF = 0.55 # Forchheimer coefficient [-]
        
        # determining threshold speed
        E = 0.1 # maximum error to Forchheimer accepted [-]
        Fo_c = E/(1-E) # critical Forchheimer number [-]
        K_ref, phi_ref = 1.01e-9, 0.35 # reference permeability [m2] and porosity [-]
        phi_max = 0.5 # max porosity in channels [-]
        K = K_ref * np.square(1-phi_ref)/np.power(phi_ref,3) \
            * np.power(phi_max,3)/np.square(1-phi_max) # Kozeny-Carman permeability [m2]
        u_bar = mu/(cF*rho*np.sqrt(K)) * Fo_c # threshold speed [m/s]
        print("Fo_c =", Fo_c, " K =", K, " u_bar =", u_bar)

        self.space_dependent_law = False

        if self.space_dependent_law is False:
            # Darcy
            lambda_1 = mu
            beta_1 = 0 *u_bar
            phi_1 = lambda a: lambda_1 + beta_1*np.sqrt(np.abs(a))
            pphi_1 = lambda a: lambda_1*a + 2/3*beta_1*np.power(np.abs(a), 1.5) # primitive of phi_1
            Phi_1 = lambda a: pphi_1(a)
            range_1 = lambda a: np.logical_and(a >= 0, a <= 1)
            
            # Forsh
            lambda_2 = mu
            beta_2 = cF*rho*np.sqrt(K) *u_bar
            phi_2 = lambda a: lambda_2 + beta_2*np.sqrt(np.abs(a))
            pphi_2 = lambda a: lambda_2*a + 2/3*beta_2*np.power(np.abs(a), 1.5) # primitive of phi_2
            Phi_2 = lambda a: pphi_2(a) + pphi_1(1) - pphi_2(1)
            range_2 = lambda a: a > 1

            # get permeability
            phi = [phi_1, phi_2]
            Phi = [Phi_1, Phi_2]
            ranges = [range_1, range_2]
            if self.region is None:
                k_ref = compute_permeability(epsilon, ranges, phi=phi, Phi=Phi)
            else:
                k_ref = lambda _: 0

            self.k_adapt = lambda flux2: k_ref(flux2) / u_bar
            self.k_darcy = lambda _: 1/lambda_1 / u_bar
            self.k_forsh = lambda flux2: 1/phi_2(flux2) / u_bar

        else:
            self.k_adapt = []
            self.k_darcy = []
            self.k_forsh = []
            for i in range(self.problem.perm[:, 0].size):
                print("cell =", i)
                
                bg_perm = self.problem.perm[i,0] * pp.DARCY # background permeability
            
                # Darcy
                lambda_1 = mu
                beta_1 = 0 *u_bar
                phi_1 = lambda a: lambda_1 + beta_1*np.sqrt(np.abs(a))
                pphi_1 = lambda a: lambda_1*a + 2/3*beta_1*np.power(np.abs(a), 1.5) # primitive of phi_1
                Phi_1 = lambda a: pphi_1(a)
                range_1 = lambda a: np.logical_and(a >= 0, a <= 1)
            
                # Forsh
                lambda_2 = mu
                beta_2 = cF*rho*np.sqrt(bg_perm) *u_bar
                phi_2 = lambda a: lambda_2 + beta_2*np.sqrt(np.abs(a))
                pphi_2 = lambda a: lambda_2*a + 2/3*beta_2*np.power(np.abs(a), 1.5) # primitive of phi_2
                Phi_2 = lambda a: pphi_2(a) + pphi_1(1) - pphi_2(1)
                range_2 = lambda a: a > 1

                # get permeability
                phi = [phi_1, phi_2]
                Phi = [Phi_1, Phi_2]
                ranges = [range_1, range_2]
                if self.region is None:
                    k_ref = compute_permeability(epsilon, phi, Phi, ranges)
                else:
                    k_ref = lambda _: 0

                self.k_adapt.append(lambda flux2: k_ref(flux2) / u_bar)
                self.k_darcy.append(lambda _: 1/lambda_1 / u_bar)
                self.k_forsh.append(lambda flux2: 1/phi_2(flux2) / u_bar)

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

        if self.space_dependent_law is False:
            if self.region is not None:
                k = np.zeros(sd.num_cells)
                # region zero is forch, region one is darcy
                darcy_region = self.region
                forsh_region = np.logical_not(self.region)

                k[darcy_region] = self.k_darcy(flux_norm2[darcy_region])
                k[forsh_region] = self.k_forsh(flux_norm2[forsh_region])
            else:
                k = self.k_adapt(flux_norm2)
        else:
            k = np.zeros(sd.num_cells)
            if self.region is not None:
                # region zero is forch, region one is darcy
                darcy_region = self.region
                forsh_region = np.logical_not(self.region)
            
                for i in range(sd.num_cells):
                    if darcy_region[i]:
                        k[i] = self.k_darcy[i](flux_norm2[i])
                    else:
                        k[i] = self.k_forsh[i](flux_norm2[i])
            else:
                for i in range(sd.num_cells):
                    k[i] = self.k_adapt[i](flux_norm2[i])

        return k * self.problem.perm[:, 0] * pp.DARCY

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
        out_flow = b_face_centers[1] > self.problem.full_physdims[1] - tol

        # define inflow type boundary conditions
        in_flow = b_face_centers[1] < 0 + tol

        # define the labels and values for the boundary faces
        labels = np.array(["neu"] * b_faces.size)
        bc_val = np.zeros(sd.num_faces)

        labels[in_flow] = "dir"
        labels[out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1.e7

        return labels, bc_val

    # ------------------------------------------------------------------------------#
