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

    def __init__(self, gb, model="flow", discr=pp.RT0):

        self.model = model
        self.gb = gb
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

        for g, d in self.gb:
            param = {}

            d["deviation_from_plane_tol"] = 1e-8
            d["is_tangential"] = True

            # assign permeability
            k = data["k"](g, d, self)

            # no source term is assumed by the user
            if g.dim == 1:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k, kyy=1, kzz=1)
            elif g.dim == 2:
                param["second_order_tensor"] = pp.SecondOrderTensor(kxx=k, kyy=k, kzz=1)
            param["source"] = data["source"](g, d, self)
            param["vector_source"] = data["vector_source"](g, d, self)

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, param["bc_values"] = data["bc"](g, data, data["tol"], self)
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                param["bc_values"] = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, np.empty(0), np.empty(0))

            pp.initialize_data(g, d, self.model, param)

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # set the discretization for the grids
        for g, d in self.gb:
            discr = self.discr(self.model)
            source = self.source(self.model)

            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.discr_name: discr,
                                                    self.source_name: source}}
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)

            # retrive the discretization of the master and slave grids
            discr_master = self.gb.node_props(g_master, pp.DISCRETIZATION)[self.variable][self.discr_name]
            discr_slave = self.gb.node_props(g_slave, pp.DISCRETIZATION)[self.variable][self.discr_name]
            coupling = self.coupling(self.model, discr_master, discr_slave)

            d[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, coupling),
                }
            }
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # assembler
        self.assembler = pp.Assembler(self.gb)
        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x, u_bar=None):
        self.assembler.distribute_variable(x)

        discr = self.discr(self.model)
        for g, d in self.gb:
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = discr.extract_pressure(g, var, d)
            d[pp.STATE][self.flux] = discr.extract_flux(g, var, d)
            d[pp.STATE][self.permeability] = np.log10(d[pp.PARAMETERS][self.model]["second_order_tensor"].values[0, 0])

            if "original_id" in g.tags:
                d[pp.STATE]["original_id"] = g.tags["original_id"] * np.ones(g.num_cells)
            if "condition" in g.tags:
                d[pp.STATE]["condition"] = g.tags["condition"] * np.ones(g.num_cells)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, discr, self.flux, self.P0_flux, self.mortar)

        for g, d in self.gb:
            norm = np.linalg.norm(d[pp.STATE][self.P0_flux], axis=0)
            d[pp.STATE][self.P0_flux_norm] = norm
            if u_bar is not None:
                d[pp.STATE][self.region] = (norm < u_bar).astype(int)

    # ------------------------------------------------------------------------------#
