import numpy as np
from scipy.stats import hmean
import porepy as pp

import sys
sys.path.insert(0, "../../src/")

from weighted_norm import *

# ------------------------------------------------------------------------------#


class Problem(object):
    def __init__(self, parameters, pos_x=None, pos_y=None):
        self.parameters = parameters

        self.full_shape = (60, 220, 85)
        self.full_physdims = (365.76, 670.56, 51.816)

        if pos_x is not None:
            self.pos_x = np.asarray(pos_x)
        else:
            self.pos_x = np.arange(self.full_shape[0])

        if pos_y is not None:
            self.pos_y = np.asarray(pos_y)
        else:
            self.pos_y = np.arange(self.full_shape[1])

        self.N = 0
        self.n = 0
        self._compute_size()

        self.mdg = None
        self._create_mdg()

        self.perm = None
        self.layers_id = None
        self.partition = None

        self._read_perm()
        self._compute_threshold_flux()

    # ------------------------------------------------------------------------------#

    def _compute_size(self):
        dim = self.parameters.layers.size

        x_range = np.linspace(0, self.full_physdims[0], self.full_shape[0] + 1)[
            np.r_[self.pos_x, self.pos_x[-1] + 1]
        ]
        y_range = np.linspace(0, self.full_physdims[1], self.full_shape[1] + 1)[
            np.r_[self.pos_y, self.pos_y[-1] + 1]
        ]

        bounding_box = {
            "xmin": x_range[0],
            "xmax": x_range[-1],
            "ymin": y_range[0],
            "ymax": y_range[-1],
            "zmin": 0,
            "zmax": 0,
        }
        if dim == 1:
            self.shape = np.hstack((self.pos_x.size, self.pos_y.size))
            self.physdims = np.hstack((x_range[-1], y_range[-1]))
        else:
            self.shape = np.hstack((self.pos_x.size, self.pos_y.size, dim))
            thickness = self.full_physdims[2] / self.full_shape[2] * dim
            self.physdims = np.hstack((x_range[-1], y_range[-1], thickness))
            bounding_box["zmax"] = self.physdims[-1]

        self.N = np.prod(self.shape)
        self.n = np.prod(self.shape[:2])
        self.domain = pp.Domain(bounding_box=bounding_box)

    # ------------------------------------------------------------------------------#

    def _create_mdg(self,):
        self.sd = pp.CartGrid(self.shape, self.physdims)
        self.sd.compute_geometry()

        # it's only one grid but the solver is build on a mdg
        self.mdg = pp.meshing.subdomains_to_mdg([self.sd])

    # ------------------------------------------------------------------------------#

    def _read_perm(self):
        shape = (self.n, self.parameters.layers.size)
        perm_xx, perm_yy, perm_zz = np.empty(shape), np.empty(shape), np.empty(shape)
        layers_id = np.empty(shape)

        for pos, layer in enumerate(self.parameters.layers):
            # get background/intrinsic perm
            perm_layer = self.parameters.perm_layer[pos]

            # fill variables to visualize
            perm_x = perm_layer[:, 0].reshape(self.full_shape[:2], order="F")
            perm_y = perm_layer[:, 1].reshape(self.full_shape[:2], order="F")
            perm_z = perm_layer[:, 2].reshape(self.full_shape[:2], order="F")

            perm_xx[:, pos] = perm_x[self.pos_x, :][:, self.pos_y].flatten(order="F")
            perm_yy[:, pos] = perm_y[self.pos_x, :][:, self.pos_y].flatten(order="F")
            perm_zz[:, pos] = perm_z[self.pos_x, :][:, self.pos_y].flatten(order="F")

            layers_id[:, pos] = layer

        # reshape
        shape = self.n * self.parameters.layers.size
        perm_xx = perm_xx.reshape(shape, order="F")
        perm_yy = perm_yy.reshape(shape, order="F")
        perm_zz = perm_zz.reshape(shape, order="F")
        self.perm = np.stack((perm_xx, perm_yy, perm_zz)).T

        self.layers_id = layers_id.reshape(shape, order="F")

    # ------------------------------------------------------------------------------#

    def _compute_threshold_flux(self):
        mu = self.parameters.mu
        c_F = self.parameters.c_F
        Fo_c = self.parameters.Fo_c
        M = self.parameters.m - 1
        diss = self.parameters.dissipative

        dim = self.sd.dim 
        u_bar_factor = 0
        if dim > 0:
            kappa = kappa_min = self.perm[:, 0]
            for j in range(1, dim):
                kappa = [k * l for k, l in zip(kappa, self.perm[:, j])]
                kappa_min = [np.minimum(k, l) for k, l in zip(kappa_min, self.perm[:, j])]
            self.kappa = np.asarray([np.power(k, 1/dim) for k in kappa])
            if diss:
                u_bar_factor = 1
            else: 
                self.kappa_min = np.asarray(kappa_min)
                u_bar_factor = np.min(np.asarray(
                    mu*np.sqrt(self.kappa_min)/(self.kappa*np.power(c_F, 1/M))
                ))

        self.u_bar = u_bar_factor * np.power(Fo_c, 1/M) # threshold flux

        if diss:
            print("u_bar =", round(self.u_bar, 5), "[-]")
        else:
            print("u_bar =", round(self.u_bar, 5), "[kg/m2/s]")

        # we also compute a refactored permeability useful in dissipative model
        gamma = np.power(c_F, 2/M) * np.square(self.kappa/mu) if diss else 1
        self.perm_diss = np.empty(self.perm.shape)
        for j in range(self.perm.shape[1]):
            self.perm_diss[:, j] = np.divide(self.perm[:, j], gamma)

    # ------------------------------------------------------------------------------#

    def save_perm(self):
        names = ["log10_perm_xx", "log10_perm_yy", "log10_perm_zz", "layer_id", 
                 "perm_xx", "perm_yy", "perm_zz"]

        # for visualization export the intrinsic perm and layer id
        for sd, d in self.mdg.subdomains(return_data=True):
            if sd.dim == self.mdg.dim_max():
                pp.set_solution_values(names[0], np.log10(self.perm[:, 0]), d, 0)
                pp.set_solution_values(names[1], np.log10(self.perm[:, 1]), d, 0)
                pp.set_solution_values(names[2], np.log10(self.perm[:, 2]), d, 0)

                pp.set_solution_values(names[3], self.layers_id, d, 0)

                pp.set_solution_values(names[4], self.perm[:, 0], d, 0)
                pp.set_solution_values(names[5], self.perm[:, 1], d, 0)
                pp.set_solution_values(names[6], self.perm[:, 2], d, 0)

        return names

    # ------------------------------------------------------------------------------#

    def perm_as_dict(self):
        return {"kxx": self.perm[:, 0], "kyy": self.perm[:, 1], "kzz": self.perm[:, 2]}

    # ------------------------------------------------------------------------------#

    def save_forch_vars(self, flux=None):
        names = [
            "Forchheimer number", \
            "P0_darcy_flux_denormalized", "P0_darcy_flux_denormalized_norm", \
            "P0_darcy_velocity", "P0_darcy_velocity_norm"]

        # for visualization export the Forchheimer number and denormalized fluxes (*u_bar)
        if flux is None:  # no flux is given: only give name of variable to visualize
            return names
        else:  # flux is given: compute Forchheimer number and store it
            rho = self.parameters.rho
            M = self.parameters.m - 1
            diss = self.parameters.dissipative
            u_bar = self.u_bar
            for sd, d in self.mdg.subdomains(return_data=True):
                flux_denorm = flux * u_bar
                norm_flux = weighted_norm(flux_denorm, self.perm_diss, sd.dim) if diss \
                    else np.linalg.norm(flux_denorm, axis=0)
                pp.set_solution_values(names[0], np.power(norm_flux, M), d, 0)
                pp.set_solution_values(names[1], flux_denorm, d, 0)
                pp.set_solution_values(names[2], norm_flux, d, 0)
                pp.set_solution_values(names[3], flux_denorm/rho, d, 0)
                pp.set_solution_values(names[4], norm_flux/rho, d, 0)
