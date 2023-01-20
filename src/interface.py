import numpy as np
import porepy as pp
import scipy.sparse as sps

import sys; sys.path.insert(0, "../../src/")
import tags

# ------------------------------------------------------------------------------#

def detect_interface(gb, network, network_original, scheme, condition_interface, flux_threshold, epsilon, tol):

    # new points to define the domains
    dom_pts = dict.fromkeys(network.tags["original_id"], np.empty((3, 0)))
    # the condition (True considition is satisfied with a <, False with a >) at each interface point
    dom_cond = dict.fromkeys(network.tags["original_id"], np.empty((2, 0)))
    # in the case of empty interface keep tracking on the condition to apply
    dom_all_Omega_1 = dict.fromkeys(network.tags["original_id"], True)
    dom_all_Omega_2 = dict.fromkeys(network.tags["original_id"], True)

    # do only this computation only for the higher dimensional grids
    for g in gb.grids_of_dimension(gb.dim_max()):
        # get the data for the current grid
        d = gb.node_props(g)
        f_id = int(np.atleast_1d(g.tags["original_id"])[0])

        # for the current grid compute the flux at each face, 1 co-dimensional grid edge included
        flux = _get_flux(g, d, gb.edges_of_node(g), scheme.flux, scheme.mortar)

        # compute the new points according to the interface + epsilon
        new_pts, cond_pts, all_Omega_1, all_Omega_2 = _has_interface(g, d, scheme, flux, condition_interface,
                                                                     flux_threshold + epsilon, tol)

        # save the information
        dom_pts[f_id] = np.hstack((dom_pts[f_id], new_pts))
        dom_cond[f_id] = np.hstack((dom_cond[f_id], cond_pts))
        dom_all_Omega_1[f_id] = dom_all_Omega_1[f_id] and all_Omega_1
        dom_all_Omega_2[f_id] = dom_all_Omega_2[f_id] and all_Omega_2

        # compute the new points according to the interface - epsilon
        new_pts, cond_pts, all_Omega_1, all_Omega_2 = _has_interface(g, d, scheme, flux, condition_interface,
                                                                     flux_threshold - epsilon, tol)

        # save the information
        dom_pts[f_id] = np.hstack((dom_pts[f_id], new_pts))
        dom_cond[f_id] = np.hstack((dom_cond[f_id], cond_pts))
        dom_all_Omega_1[f_id] = dom_all_Omega_1[f_id] and all_Omega_1
        dom_all_Omega_2[f_id] = dom_all_Omega_2[f_id] and all_Omega_2

    new_network_pts = np.empty((3, 0))
    new_network_edges = np.empty((2, 0), dtype=np.int)
    new_network_tag_original_id = np.empty((0), dtype=np.int)
    new_network_tag_condition = np.empty((0), dtype=np.int)
    num_pts = 0

    # for each orginal id order the points
    for f_id in np.unique(network.tags["original_id"]):

        # consider also the points from the original network
        mask_original = network_original.tags["original_id"] == f_id
        edges_original = network_original.edges[:, mask_original].ravel()
        pts_original = network_original.pts[:, np.unique(edges_original)]

        if pts_original.shape[1] == 2:
            pts_original = np.vstack((pts_original, np.zeros(pts_original.shape[0])))

        # add the original pts to the list
        all_pts = np.hstack((dom_pts[f_id], pts_original))
        # we add to the condition a NONE for the points in the original network
        cond = np.hstack((dom_cond[f_id], [[tags.NONE, tags.NONE]]*pts_original.shape[1]))

        # we need to order the points now, we assume again that we are on a line
        R = pp.map_geometry.project_line_matrix(all_pts, tol=1e-5)
        pts = np.dot(R, all_pts)

        # determine which dimension is active
        check = np.sum(np.abs(pts.T - pts[:, 0]), axis=0)
        dim = np.logical_not(np.isclose(check/np.sum(check), 0, atol=1e-5, rtol=0))
        mask = np.argsort(pts[dim, :].ravel())

        # order the points
        all_pts = all_pts[:, mask]

        # the order is flipped, we need to restore the order coherent with the original pts
        if not np.allclose(all_pts[:, 0], pts_original[:, 0]):
            all_pts = np.fliplr(all_pts)
            mask = mask[::-1]

        cond = cond[:, mask]

        import pdb; pdb.set_trace()
        # we need to flip the condition, this works for at least two interfaces
        if np.any(np.sort(mask[1:-1]) != mask[1:-1]):
            cond = np.flipud(cond)

        print("-----")
        print(cond)
        print(cond.shape)
        # consider now that the interfaces are computed as flux_threshold + and - epsilon
        for i in np.arange(cond.shape[1]-1):
            if np.all(cond[:, i] == cond[:, i+1]):
                cond[1, i] = tags.INTERFACE
                cond[0, i+1] = tags.INTERFACE
        print("=====")
        print(cond)
        print(cond.shape)

        # compute the edge condition from the point condition
        mask = np.where(cond[0, :] == tags.NONE)[0]
        # remove the last point, the condition is given by the "left" n-1 point
        mask = np.setdiff1d(mask, cond.shape[1] - 1)
        # inherit the condition from the left
        cond[:, mask] = cond[0, mask+1]

        print("+++++")
        print(cond)
        print(cond.shape)

        # special case where only the original points are present and no new internal interface
        if cond.shape[1] == 2:
            if dom_all_Omega_1[f_id]:
                cond[1, 0] = tags.OMEGA_1
            elif dom_all_Omega_2[f_id]:
                cond[1, 0] = tags.OMEGA_2

        import pdb; pdb.set_trace()


        # set the condition for the edges
        new_network_tag_condition = np.hstack((new_network_tag_condition, cond[1, :-1]))

        # we assume no duplicate pts
        # pts, new_2_old, old_2_new = pp.utils.setmembership.unique_columns_tol(pts, tol=tol)
        # construct the edges
        edges = np.empty((2, pts.shape[1]-1), dtype=np.int)
        edges[0, :] = np.arange(edges.shape[1]) + num_pts
        edges[1, :] = edges[0, :] + 1
        num_pts += pts.shape[1]

        # collect all the infromations
        new_network_pts = np.hstack((new_network_pts, all_pts))
        new_network_edges = np.hstack((new_network_edges, edges))
        new_network_tag_original_id = np.hstack((new_network_tag_original_id,
                                                 f_id * np.ones(edges.shape[1], dtype=np.int)))

    # create the new network
    new_network = pp.FractureNetwork2d(new_network_pts[:2, :], new_network_edges, network.domain)
    new_network.tags["original_id"] = new_network_tag_original_id
    new_network.tags["condition"] = new_network_tag_condition

    return new_network

# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#

def _get_flux(g, d, g_edges, flux_name, mortar_name):
    # we need to recover the flux from the mortar variable
    # only lower dimensional edges need to be considered.
    flux = d[pp.STATE][flux_name]
    if np.any(g.tags["fracture_faces"]):
        # recover the sign of the flux, since the mortar is assumed
        # to point from the higher to the lower dimensional problem
        _, indices = np.unique(g.cell_faces.indices, return_index=True)
        sign = sps.diags(g.cell_faces.data[indices], 0)

        for _, d_e in g_edges:
            g_m = d_e["mortar_grid"]
            if g_m.dim == g.dim:
                continue
            # project the mortar variable back to the higher dimensional
            # problem
            flux += (
                sign * g_m.primary_to_mortar_avg().T * d_e[pp.STATE][mortar_name]
            )

    return flux

# ------------------------------------------------------------------------------#

def _has_interface(g, d, scheme, flux, condition_interface, flux_threshold,
                   evaluation_tol=1e-5, evaluation_max_iter=100):
    discr = d[pp.DISCRETIZATION][scheme.variable][scheme.discr_name]

    # detect if the condition is met, since the velocity is linear then we first check in the
    # boundaries of each grid element
    faces, cells, sign = sps.find(g.cell_faces)
    index = np.argsort(cells)
    faces, sign = faces[index], sign[index]

    # Map the domain to a reference geometry (i.e. equivalent to compute
    # surface coordinates in 1d and 2d)
    R = pp.map_geometry.project_line_matrix(g.nodes, tol=1e-5)
    f_centers = np.dot(R, g.face_centers)
    # check if the ordering is flip due to the rotation matrix
    #is_flip = np.sign(np.arctan2(R[2, 1], R[2, 2])) ## old version
    is_flip = np.sign(np.arccos((np.trace(R)-1)/2.))
    node_coords = np.dot(R, g.nodes)

    # determine which dimension is active
    check = np.sum(np.abs(f_centers.T - f_centers[:, 0]), axis=0)
    dim = np.logical_not(np.isclose(check/np.sum(check), 0, atol=1e-5, rtol=0))

    # compute the note to translate the pts afterwards
    node_t = np.zeros((3, 1))
    mask = np.logical_not(dim)
    node_t[mask, :] = f_centers[mask, 0].reshape((np.sum(mask), -1))

    # map the geometrical data
    node_coords = (node_coords - node_t)[dim, :]
    f_centers = (f_centers - node_t)[dim, :]
    f_normals = np.dot(R, g.face_normals)[dim, :]

    # get the opposite face-nodes
    cell_face_to_opposite_node = d[discr.cell_face_to_opposite_node]

    # list of grid points + interface points with the condition intersection label
    interface_pts = np.empty((f_centers.shape[0]+1, 0))
    # for each interface point we store if, in the local coordinate system, the condition is true or false
    # on the left and on the right. The value of tags.OMEGA_1 means the condition is satisfied with a <, while with
    # a tags.OMEGA_2 the condition is satisfied with a >, and with tags.INTERFACE the condition is satisfied with a
    # ==
    interface_cond = np.empty((2, 0), dtype=np.int)

    # consider if all the faces have condition less or more the interface condition
    all_Omega_1 = True
    all_Omega_2 = True

    for c in np.arange(g.num_cells):
        # For the current cell retrieve its faces
        loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
        faces_loc = faces[loc]

        # get the opposite node id for each face
        node = cell_face_to_opposite_node[c, :]
        coord_loc = node_coords[:, node]

        # check what's happening in the cell-boundary, since the flux is maximum there
        check = np.zeros(faces_loc.size, dtype=np.int)

        for idx, face_loc in enumerate(faces_loc):
            P = scheme.discr.faces_to_cell(
                    f_centers[:, face_loc],
                    coord_loc,
                    f_centers[:, faces_loc],
                    f_normals[:, faces_loc],
                    dim,
                    R,
                )
            check[idx] = condition_interface(flux_threshold, np.dot(P, flux[faces_loc]), evaluation_tol)

        all_Omega_1 = all_Omega_1 and np.all(check == tags.OMEGA_1)
        all_Omega_2 = all_Omega_2 and np.all(check == tags.OMEGA_2)

        if not(np.all(check == tags.OMEGA_1) or np.all(check == tags.OMEGA_2)):
            if np.any(check == tags.INTERFACE):
                probe_pt = f_centers[:, faces_loc[check == tags.INTERFACE]]

            # if the condition is valid for a cell-boundary we might need to compute internally
            #if not(np.all(check == tags.OMEGA_1) or np.all(check == tags.OMEGA_2)) and np.any(check == tags.OMEGA_1):
            elif np.sum(check == tags.OMEGA_1) == 1 and np.sum(check == tags.OMEGA_2) == 1:
                #NOTE: this algorithm may work only in 1d
                bd_pts = f_centers[:, faces_loc]
                not_okay = True
                it = 0
                while not_okay and it < evaluation_max_iter:
                    # bisection method to get the actual value of the interface
                    probe_pt = np.average(bd_pts, axis=1)
                    P = scheme.discr.faces_to_cell(
                            probe_pt,
                            coord_loc,
                            f_centers[:, faces_loc],
                            f_normals[:, faces_loc],
                            dim,
                            R,
                        )
                    if condition_interface(flux_threshold, np.dot(P, flux[faces_loc]), evaluation_tol) == tags.INTERFACE:
                        not_okay = False
                    else:
                        cond = condition_interface(flux_threshold, np.dot(P, flux[faces_loc]), evaluation_tol)
                        bd_pts[:, check == cond] = probe_pt
                    it += 1
                if it == evaluation_max_iter:
                    raise ValueError

            # store the interface point
            interface_pts = np.hstack((interface_pts, np.vstack((probe_pt, True))))
            # check the ordering
            supposed_ordering = np.array([1, -1])*is_flip

            diff = probe_pt - f_centers[:, faces_loc]
            is_not_zero = np.logical_not(np.abs(diff) < evaluation_tol)

            #ordering = np.all(np.equal(np.sign(diff), supposed_ordering))
            ordering = np.all(np.equal(np.sign(diff[is_not_zero]), supposed_ordering[is_not_zero.ravel()]))

            check = check if ordering else np.flipud(check)
            interface_cond = np.hstack((interface_cond, np.atleast_2d(check).T))

    # map back the interface points since the mapping is local to the grid
    pts = np.zeros((3, interface_pts.shape[1]))
    pts[dim] = interface_pts[0]

    print(pts, interface_cond)

    pts, _, old_2_new = pp.utils.setmembership.unique_columns_tol(pts, evaluation_tol)

    # in case of pts merged
    if not np.all(old_2_new == np.arange(interface_cond.shape[1])):
        cond = np.empty((2, pts.shape[1]))
        for idx in np.unique(old_2_new):
            cd = interface_cond[:, old_2_new == idx]
            cond[0, idx] = cd[0, 0]
            cond[1, idx] = cd[1, 1]
        interface_cond = cond

    return np.dot(R.T, pts + node_t), interface_cond, all_Omega_1, all_Omega_2

# ------------------------------------------------------------------------------#
