import numpy as np
import scipy.sparse as sps

import porepy as pp

class DualEllipticCodim2(pp.EllipticDiscretization):
    def __init__(self, keyword):
        super(DualEllipticCodim2, self).__init__(keyword)

    def assemble_int_bound_pressure_trace_codim_2(
        self,
        sd,
        intf,
        data,
        data_intf,
        cc,
        matrix,
        rhs,
        self_ind,
        use_secondary_proj=False,
        assemble_matrix=True,
        assemble_rhs=True,
    ):
        """Assemble the contribution from an internal
        boundary, manifested as a condition on the boundary pressure.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            sd (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_intf (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]

        if use_secondary_proj:
            proj = intf.secondary_to_mortar_avg()
            proj_int = intf.mortar_to_secondary_int()
        else:
            proj = intf.primary_to_mortar_avg()
            if assemble_matrix:
                proj_int = intf.mortar_to_primary_int()

        if assemble_matrix:
            shape = (proj.shape[0], sd.num_faces)
            M = sps.diags(np.ones(sd.num_cells), format="csc")
            cc[2, self_ind] += sps.bmat([[sps.csr_matrix(shape), proj @ M]])

            #cc[2, 2] += proj * M * proj_int DA CONTROLLARE, LA M POTEBBE NON ESSERE QUELLA GIUSTA
        ### Add contribution from boundary conditions to the pressure at the fracture
        ### faces. For TPFA this will be zero, but for MPFA we will get a contribution
        ### on the fractures extending to the boundary due to the interaction region
        ### around a node.
        ##if assemble_rhs:
        ##    bc_val = parameter_dictionary["bc_values"]
        ##    rhs[2] -= (
        ##        proj * matrix_dictionary[self.bound_pressure_face_matrix_key] * bc_val
        ##    )

        ##    # Add gravity contribution if relevant
        ##    if "vector_source" in parameter_dictionary:
        ##        vector_source_discr = matrix_dictionary[
        ##            self.bound_pressure_vector_source_matrix_key
        ##        ]
        ##        # The vector source, defaults to zero if not specified.
        ##        vector_source = parameter_dictionary.get("vector_source")
        ##        rhs[2] -= proj * vector_source_discr * vector_source

    def assemble_int_bound_flux_codim_2(
        self, sd, intf, data, data_intf, cc, matrix, rhs, self_ind, use_secondary_proj=False
    ):
        """Assemble the contribution from an internal boundary, manifested as a
        flux boundary condition. TO FIX THE TEXT

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a higher-dimensional domain.

        Implementations of this method will use an interplay between the grid
        on the node and the mortar grid on the relevant edge.

        Parameters:
            sd (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_intf (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                primary and secondary side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.
            use_secondary_proj (boolean): If True, the secondary side projection operator is
                used. Needed for periodic boundary conditions.

        """
        # Projection operators to grid
        if use_secondary_proj:
            proj = intf.mortar_to_secondary_int()
        else:
            proj = intf.mortar_to_primary_int()

        shape = (sd.num_faces, proj.shape[1])
        cc[self_ind, 2] += sps.bmat([[sps.csr_matrix(shape)], [proj]])

class RT0Codim2(pp.RT0, DualEllipticCodim2):
    def __init__(self, keyword):
        super(RT0Codim2, self).__init__(keyword)

class MVEMCodim2(pp.MVEM, DualEllipticCodim2):
    def __init__(self, keyword):
        super(MVEMCodim2, self).__init__(keyword)
