from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
import porepy.numerics.interface_laws.abstract_interface_law

module_sections = ["numerics"]

class WellCoupling(
    porepy.numerics.interface_laws.abstract_interface_law.AbstractInterfaceLaw
):
    """Simple version of the classical Peaceman model relating electrode and fracture
    pressure.

    """

    def __init__(
        self,
        keyword: str,
        discr_primary: Optional["pp.EllipticDiscretization"] = None,
        discr_secondary: Optional["pp.EllipticDiscretization"] = None,
        primary_keyword: Optional[str] = None,
    ) -> None:
        """Initialize Robin Coupling.

        Parameters:
            keyword (str): Keyword used to access parameters needed for this
                discretization in the data dictionary. Will also define where
                 discretization matrices are stored.
            discr_primary: Discretization on the higher-dimensional neighbor. Only
                needed when the RobinCoupling is used for local assembly.
            discr_secondary: Discretization on the lower-dimensional neighbor. Only
                needed when the RobinCoupling is used for local assembly. If not
                provided, it is assumed that primary and secondary discretizations
                are identical.
            primary_keyword: Parameter keyword for the discretization on the higher-
                dimensional neighbor, which the RobinCoupling is intended coupled to.
                Only needed when the object is not used for local assembly (that is,
                needed if Ad is used).
        """
        super().__init__(keyword)

        if discr_secondary is None:
            discr_secondary = discr_primary

        if primary_keyword is not None:
            self._primary_keyword = primary_keyword
        else:
            if discr_primary is None:
                raise ValueError(
                    "Either primary keyword or primary discretization must be specified"
                )
            else:
                self._primary_keyword = discr_primary.keyword

        self.discr_primary = discr_primary
        self.discr_secondary = discr_secondary

        # Keys used to identify the discretization matrices of this discretization
        self.well_discr_matrix_key: str = "well_mortar_discr"
        self.well_vector_source_matrix_key: str = "well_vector_source_discr"

    def ndof(self, intf: pp.MortarGrid) -> int:
        return intf.num_cells

    def discretize(
        self, sd_primary: pp.Grid, sd_secondary: pp.Grid, intf: pp.MortarGrid, data_primary: Dict, data_secondary: Dict, data_intf: Dict
    ) -> None:
        """Discretize the Peaceman interface law and store the discretization in the
        edge data.

        Parameters:
            sd_primary: Grid of the primary domanin.
            sd_secondary: Grid of the secondary domain.
            data_primary: Data dictionary for the primary domain.
            data_secondary: Data dictionary for the secondary domain.
            data_intf: Data dictionary for the edge between the domains.

        Implementational note: The computation of equivalent radius is highly simplified
        and ignores discretization and anisotropy effects. For more advanced alternatives,
        see the MRST book, https://www.cambridge.org/core/books/an-introduction-to-
        reservoir-simulation-using-matlabgnu-octave/F48C3D8C88A3F67E4D97D4E16970F894
        """
        matrix_dictionary_edge: Dict[str, sps.spmatrix] = data_intf[
            pp.DISCRETIZATION_MATRICES
        ][self.keyword]
        parameter_dictionary_edge: Dict = data_intf[pp.PARAMETERS][self.keyword]

        parameter_dictionary_h: Dict = data_primary[pp.PARAMETERS][self._primary_keyword]
        parameter_dictionary_l: Dict = data_secondary[pp.PARAMETERS][self.keyword]
        # projection matrix
        proj_h = intf.primary_to_mortar_avg()
        proj_l = intf.secondary_to_mortar_avg()

        r_w = parameter_dictionary_l["well_radius"]
        skin_factor = parameter_dictionary_edge["skin_factor"]
        # Compute equivalent radius for Peaceman well model (see above note)
        r_e = 0.2 * np.power(sd_primary.cell_volumes, 1 / sd_primary.dim)
        # Compute effective permeability
        k: pp.SecondOrderTensor = parameter_dictionary_h["second_order_tensor"]
        if sd_primary.dim == 2:
            R = pp.map_geometry.project_plane_matrix(sd_primary.nodes)
            k = k.copy()
            k.values = np.tensordot(R.T, np.tensordot(R, k.values, (1, 0)), (0, 1))
            k.values = np.delete(k.values, (2), axis=0)
            k.values = np.delete(k.values, (2), axis=1)
        kx = k.values[0, 0]
        ky = k.values[1, 1]
        ke = np.sqrt(kx * ky)

        # Kh is ke * specific volume, but this is already captured by k
        Kh = proj_h * ke * intf.cell_volumes
        WI = sps.diags(
            2 * np.pi * Kh / (np.log(proj_h * r_e / r_w) + skin_factor), format="csr"
        )
        matrix_dictionary_edge[self.well_discr_matrix_key] = -WI

        ## Vector source.
        # This contribution is last term of
        # lambda = -\int{\kappa_n [p_l - p_h +  a/2 g \cdot n]} dV,
        # where n is the outwards normal and the integral is taken over the mortar cell.
        # (Note: This assumes a P0 discretization of mortar fluxes).

        # Ambient dimension of the problem, as specified for the higher-dimensional
        # neighbor.
        # IMPLEMENTATION NOTE: The default value is needed to avoid that
        # ambient_dimension becomes a required parameter. If neither ambient dimension,
        # nor the actual vector_source is specified, there will be no problems (in the
        # assembly, a zero vector soucre of a size that fits with the discretization is
        # created). If a vector_source is specified, but the ambient dimension is not,
        # a dimension mismatch will result unless the ambient dimension implied by
        # the size of the vector source matches sd_primary.dim. This is okay for domains with
        # no subdomains with co-dimension more than 1, but will fail for fracture
        # intersections. The default value is thus the least bad option in this case.
        vector_source_dim: int = parameter_dictionary_h.get(
            "ambient_dimension", sd_primary.dim
        )
        # The ambient dimension cannot be less than the dimension of sd_primary.
        # If this is broken, we risk ending up with zero normal vectors below, so it is
        # better to break this off now
        if vector_source_dim < sd_primary.dim:
            raise ValueError(
                "Ambient dimension cannot be lower than the grid dimension"
            )

        # Construct the dot product between vectors connecting fracture and well
        # cell centers and the identity matrix.

        # Project the vectors, we need to do some transposes to get this right.
        # Note that in the codim 2 case, proj_h maps from sd_primary cells, not faces.
        vectors_mortar = (proj_h * sd_primary.cell_centers.T - proj_l * sd_secondary.cell_centers.T).T

        # The values in vals are sorted by the mortar cell index ordering (proj is a
        # csr matrix).
        ci_mortar = np.arange(intf.num_cells, dtype=int)

        # The mortar cell indices are expanded to account for the vector source
        # having multiple dimensions
        rows = np.tile(ci_mortar, (vector_source_dim, 1)).ravel("F")
        # Columns must account for the values being vector values.
        cols = pp.fvutils.expand_indices_nd(ci_mortar, vector_source_dim)
        vals = vectors_mortar[:vector_source_dim].ravel("F")
        # And we have the normal vectors
        expanded_vectors = sps.coo_matrix((vals, (rows, cols))).tocsr()

        # On assembly, the outwards normals on the mortars will be multiplied by the
        # interface vector source.
        matrix_dictionary_edge[self.well_vector_source_matrix_key] = expanded_vectors

    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
        matrix: sps.spmatrix,
    ):
        """AAAAAAAAA
        """
        return self._assemble(
            sd_primary,
            sd_secondary,
            intf,
            data_primary,
            data_secondary,
            data_intf,
            matrix,
        )

    def _assemble(
        self,
        sd_primary,
        sd_secondary,
        intf: pp.MortarGrid,
        data_primary,
        data_secondary,
        data_intf,
        matrix,
        assemble_matrix=True,
        assemble_rhs=True,
    ):
        """Actual implementation of assembly. May skip matrix and rhs if specified."""

        matrix_dictionary_edge = data_intf[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary_edge = data_intf[pp.PARAMETERS][self.keyword]

        primary_ind = 0
        secondary_ind = 1

        if assemble_rhs and not assemble_matrix:
            # We need not make the cc matrix to assemble local matrix contributions
            rhs = self._define_local_block_matrix(
                sd_primary,
                sd_secondary,
                intf,
                self.discr_primary,
                self.discr_secondary,
                matrix,
                create_matrix=False,
            )
            # We will occationally need the variable cc, but it need not have a value
            cc = None
        else:
            cc, rhs = self._define_local_block_matrix(
                sd_primary,
                sd_secondary,
                intf,
                self.discr_primary,
                self.discr_secondary,
                matrix,
            )

        # The convention, for now, is to put the higher dimensional information
        # in the first column and row in matrix, lower-dimensional in the second
        # and mortar variables in the third
        if assemble_matrix:
            cc[2, 2] = matrix_dictionary_edge[self.well_discr_matrix_key]

        # Assembly of contribution from boundary pressure must be called even if only
        # matrix or rhs must be assembled.
        self.discr_primary.assemble_int_bound_pressure_trace_codim_2(
            sd_primary,
            intf,
            data_primary,
            data_intf,
            cc,
            matrix,
            rhs,
            primary_ind,
            assemble_matrix=assemble_matrix,
            assemble_rhs=assemble_rhs,
        )

        if assemble_matrix:
            # Calls only for matrix assembly
            self.discr_primary.assemble_int_bound_flux_codim_2(
                sd_primary, intf, data_primary, data_intf, cc, matrix, rhs, primary_ind
            )
            self.discr_secondary.assemble_int_bound_pressure_cell(
                sd_secondary, data_secondary, intf, data_intf, cc, matrix, rhs, secondary_ind
            )
            self.discr_secondary.assemble_int_bound_source(
                sd_secondary, data_secondary, intf, data_intf, cc, matrix, rhs, secondary_ind
            )

        ####if assemble_rhs:
        ####    # Calls only for rhs assembly
        ####    if "vector_source" in parameter_dictionary_edge:
        ####        # Also assemble vector sources.
        ####        # Discretization of the vector source term
        ####        vector_source_discr: sps.spmatrix = matrix_dictionary_edge[
        ####            self.mortar_vector_source_matrix_key
        ####        ]
        ####        # The vector source, defaults to zero if not specified.
        ####        vector_source: np.ndarray = parameter_dictionary_edge.get(
        ####            "vector_source"
        ####        )
        ####        if vector_source_discr.shape[1] != vector_source.size:
        ####            # If this happens chances are that either the ambient dimension was not set
        ####            # and thereby its default value was used. Another not unlikely reason is
        ####            # that the ambient dimension is set, but with a value that does not match
        ####            # the specified vector source.
        ####            raise ValueError(
        ####                """Mismatch in vector source dimensions.
        ####                Did you forget to specify the ambient dimension?"""
        ####            )

        ####        rhs[2] = rhs[2] - vector_source_discr * vector_source

        ####    rhs[2] = matrix_dictionary_edge[self.mortar_scaling_matrix_key] * rhs[2]

        if assemble_matrix:
        ###    for block in range(cc.shape[1]):
        ###        # Scale the pressure blocks in the mortar problem
        ###        import pdb; pdb.set_trace()
        ###        cc[2, block] = (
        ###            matrix_dictionary_edge[self.mortar_scaling_matrix_key]
        ###            * cc[2, block]
        ###        )
            matrix += cc

        ###    self.discr_primary.enforce_neumann_int_bound(
        ###        sd_primary, data_intf, matrix, primary_ind
        ###    )

        if not assemble_matrix:
            return rhs
        elif not assemble_rhs:
            return matrix
        else:
            return matrix, rhs

    def __repr__(self) -> str:
        return f"Interface coupling of Well type, with keyword {self.keyword}"
