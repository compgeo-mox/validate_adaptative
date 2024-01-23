import numpy as np
import porepy as pp

from well_coupling import WellCoupling
from weighted_norm import *


class Flow(object):
    # post process variables
    pressure = "pressure"
    flux = "darcy_flux"  # it has to be this one
    P0_flux = "P0_darcy_flux"
    P0_flux_norm = "P0_darcy_flux_norm"
    region = "region"
    permeability = "permeability"
    gradient_pressure = "gradient_pressure"

    # ------------------------------------------------------------------------------#

    def __init__(self, mdg, model="flow", discr=pp.RT0):
        self.model = model
        self.mdg = mdg
        self.data = None
        self.data_time = None
        self.assembler = None
        self.assembler_variable = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = discr

        # coupling operator
        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.FluxPressureContinuity

        # source
        self.source_name = self.model + "_source"
        self.source = pp.DualScalarSource

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar = self.model + "_lambda"

    # ------------------------------------------------------------------------------#

    def set_data(self, data):
        self.data = data

        for sd, d in self.mdg.subdomains(return_data=True):
            param = {}

            d["deviation_from_plane_tol"] = 1e-8
            d["is_tangential"] = True
            d["ambient_dimension"] = (self.mdg.dim_max(),)

            # assign permeability
            k = self.data["k"](sd, d, self)

            # no source term is assumed by the user
            if sd.dim == 1 and sd.well_num == -1:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k[0], kyy=1, kzz=1)
            elif sd.dim == 2:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k[0], kyy=k[1], kzz=1)
            elif sd.dim == 3:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k[0], kyy=k[1], kzz=k[2])
            elif sd.dim == 1 and sd.well_num > -1:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k[0], kyy=1, kzz=1)
                param["well_radius"] = data["well_radius"]

            param["source"] = data["source"](sd, d, self)
            param["vector_source"] = data["vector_source"](sd, d, self)

            # Boundaries
            b_faces = sd.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, param["bc_values"] = data["bc"](sd, data, data["tol"], self)
                param["bc"] = pp.BoundaryCondition(sd, b_faces, labels)
            else:
                param["bc_values"] = np.zeros(sd.num_faces)
                param["bc"] = pp.BoundaryCondition(sd, np.empty(0), np.empty(0))

            pp.initialize_data(sd, d, self.model, param)

        for intf, data in self.mdg.interfaces(return_data=True):
            _, sd_secondary = self.mdg.interface_to_subdomain_pair(intf)

            if sd_secondary.dim == 1 and sd_secondary.well_num > -1:
                param = {"skin_factor": np.zeros(intf.num_cells)}

            pp.initialize_data(intf, data, self.model, param)

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):
        discr = self.discr(self.model)
        source = self.source(self.model)

        # set the discretization for the grids
        for sd, d in self.mdg.subdomains(return_data=True):
            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {
                self.variable: {self.discr_name: discr, self.source_name: source}
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # define the interface terms to couple the grids
        coupling_well = WellCoupling(self.model, discr, discr)

        for intf, data in self.mdg.interfaces(return_data=True):
            sd_primary, sd_secondary = self.mdg.interface_to_subdomain_pair(intf)
            data[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}

            if sd_secondary.dim == 1 and sd_secondary.well_num > -1:
                coupling = coupling_well
            else:
                raise ValueError

            data[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    sd_secondary: (self.variable, self.discr_name),
                    sd_primary: (self.variable, self.discr_name),
                    intf: (self.mortar, coupling),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # assembler
        self.assembler = pp.Assembler(self.mdg)
        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x, u_bar=None):
        perm_diss = self.data["perm_diss"]
        diss = self.data["dissipative"]

        self.assembler.distribute_variable(x)

        discr = self.discr(self.model)
        for sd, data in self.mdg.subdomains(return_data=True):
            var = data[pp.TIME_STEP_SOLUTIONS][self.variable][0]
            pressure = discr.extract_pressure(sd, var, data)
            flux = discr.extract_flux(sd, var, data)
            k = data[pp.PARAMETERS][self.model]["second_order_tensor"].values[0, 0]

            pp.set_solution_values(self.pressure, pressure, data, 0)
            pp.set_solution_values(self.flux, flux, data, 0)
            pp.set_solution_values(self.permeability, k, data, 0)

            if "original_id" in sd.tags:
                original_id = sd.tags["original_id"] * np.ones(sd.num_cells)
                pp.set_solution_values("original_id", original_id, data, 0)

            if "condition" in sd.tags:
                condition = sd.tags["condition"] * np.ones(sd.num_cells)
                pp.set_solution_values("condition", condition, data, 0)

        # export the P0 flux reconstruction
        pp.project_flux(self.mdg, discr, self.flux, self.P0_flux, self.mortar)

        for sd, data in self.mdg.subdomains(return_data=True):
            flux = data[pp.TIME_STEP_SOLUTIONS][self.P0_flux][0]
            norm = weighted_norm(flux, perm_diss, sd.dim) if diss else np.linalg.norm(flux, axis=0)
            pp.set_solution_values(self.P0_flux_norm, norm, data, 0)

            k = data[pp.PARAMETERS][self.model]["second_order_tensor"].values[0, 0]
            gradient = -flux / k
            pp.set_solution_values(self.gradient_pressure, gradient, data, 0)

            if u_bar is not None:
                where_u_bar = (norm < u_bar).astype(int)
                pp.set_solution_values(self.region, where_u_bar, data, 0)

    # ------------------------------------------------------------------------------#
