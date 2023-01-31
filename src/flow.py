import numpy as np
import porepy as pp

class Flow(object):

    # post process variables
    pressure = "pressure"
    flux = "darcy_flux"  # it has to be this one
    P0_flux = "P0_darcy_flux"
    P0_flux_norm = "P0_darcy_flux_norm"
    region = "region"
    permeability = "permeability"

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

            # assign permeability
            k = data["k"](sd, d, self)

            # no source term is assumed by the user
            if sd.dim == 1:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k, kyy=1, kzz=1)
            elif sd.dim == 2:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k, kyy=k, kzz=1)
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

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # set the discretization for the grids
        for sd, d in self.mdg.subdomains(return_data=True):
            discr = self.discr(self.model)
            source = self.source(self.model)

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.discr_name: discr,
                                                    self.source_name: source}}
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # define the interface terms to couple the grids
        for e, d in self.mdg.interfaces(return_data=True):
            gl, gh = mdg.interface_to_subdomain_pair(e)

            # retrive the discretization of the master and slave grids
            discr_master = self.mdg.node_props(gh, pp.DISCRETIZATION)[self.variable][self.discr_name]
            discr_slave = self.mdg.node_props(gl, pp.DISCRETIZATION)[self.variable][self.discr_name]
            coupling = self.coupling(self.model, discr_master, discr_slave)

            d[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    gl: (self.variable, self.discr_name),
                    gh: (self.variable, self.discr_name),
                    e: (self.mortar, coupling),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # assembler
        self.assembler = pp.Assembler(self.mdg)
        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x, u_bar=None):
        self.assembler.distribute_variable(x)

        discr = self.discr(self.model)
        for sd, d in self.mdg.subdomains(return_data=True):
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = discr.extract_pressure(sd, var, d)
            d[pp.STATE][self.flux] = discr.extract_flux(sd, var, d)
            d[pp.STATE][self.permeability] = np.log10(d[pp.PARAMETERS][self.model]["second_order_tensor"].values[0, 0])

            if "original_id" in sd.tags:
                d[pp.STATE]["original_id"] = sd.tags["original_id"] * np.ones(sd.num_cells)
            if "condition" in sd.tags:
                d[pp.STATE]["condition"] = sd.tags["condition"] * np.ones(sd.num_cells)

        # export the P0 flux reconstruction
        pp.project_flux(self.mdg, discr, self.flux, self.P0_flux, self.mortar)

        for _, d in self.mdg.subdomains(return_data=True):
            norm = np.linalg.norm(d[pp.STATE][self.P0_flux], axis=0)
            d[pp.STATE][self.P0_flux_norm] = norm
            if u_bar is not None:
                d[pp.STATE][self.region] = (norm < u_bar).astype(int)

    # ------------------------------------------------------------------------------#
