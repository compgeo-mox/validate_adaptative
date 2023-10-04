import numpy as np
import porepy as pp
import scipy.sparse as sps

from data import Data
import sys

sys.path.insert(0, "examples/case1/")
from problem import Problem
from parameters import Parameters

import sys

sys.path.insert(0, "src/")
from flow import Flow
from exporter import write_network_pvd
from solver_codim2 import MVEMCodim2
from compute_error import compute_error

# ------------------------------------------------------------------------------#


def add_wells(mdg, domain, well_coords, well_num_cells, tol):
    # define the wells
    wells = np.array([pp.Well(e) for e in well_coords])

    # the mesh size is determined by the lenght of the wells and the imposed num_cells
    wells_length = np.array([np.linalg.norm(e[:, 1] - e[:, 0]) for e in well_coords])
    mesh_size = np.amin(wells_length / well_num_cells)

    # create the wells grids
    wells_set = pp.WellNetwork3d(domain, wells, parameters={"mesh_size": mesh_size})

    # add the wells to the MixedDimensionalGrid
    wells_set.mesh(mdg)

    # we assume only one single higher dimensional grid
    sd = mdg.subdomains(dim=3)[0]
    sd_cell_nodes = sd.cell_nodes()

    # select only some of the nodes that are on the surface
    selected_nodes = sd.nodes[2, :] >= np.amin(well_coords[:, 2]) - tol
    cells = np.flatnonzero(sd_cell_nodes.T * selected_nodes)

    # compute the intersection between the wells and the grid
    pp.fracs.wells_3d.compute_well_rock_matrix_intersections(mdg, cells, tol=tol)

    return mdg


def main(region):
    parameters = Parameters(
        layers=np.arange(25), perm_folder="examples/case1/spe10_perm/"
    )  # get parameters, print them and read porosity
    problem = Problem(
        parameters, pos_x=np.arange(20), pos_y=np.arange(20)
    )  # create the grid bucket and get intrinsic permeability
    data = Data(
        parameters, problem
    )  # get data for computation of speed-dependent permeability

    # tolerance in the computation
    tol = 1e-10

    file_name = "case4"
    folder_name = "examples/case4/fixed/"
    variable_to_export = [
        Flow.pressure,
        Flow.P0_flux,
        Flow.permeability,
        Flow.P0_flux_norm,
        Flow.region,
    ]

    max_iteration_non_linear = 20
    max_err_non_linear = 1e-4

    hx, hy, hz = problem.physdims / problem.shape

    # add the wells
    well_hight = 6.5
    well_coords = np.asarray(
        [
            np.array(
                [
                    [0.5 * hx, 0.5 * hx],
                    [0.5 * hy, 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_hight * hz],
                ]
            ),
            np.array(
                [
                    [problem.physdims[0] - 0.5 * hx, problem.physdims[0] - 0.5 * hx],
                    [0.5 * hy, 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_hight * hz],
                ]
            ),
            np.array(
                [
                    [0.5 * hx, 0.5 * hx],
                    [problem.physdims[1] - 0.5 * hy, problem.physdims[1] - 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_hight * hz],
                ]
            ),
            np.array(
                [
                    [problem.physdims[0] - 0.5 * hx, problem.physdims[0] - 0.5 * hx],
                    [problem.physdims[1] - 0.5 * hy, problem.physdims[1] - 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_hight * hz],
                ]
            ),
            np.array(
                [
                    [
                        0.5 * problem.physdims[0] - 0.5 * hx,
                        0.5 * problem.physdims[0] - 0.5 * hx,
                    ],
                    [
                        0.5 * problem.physdims[1] - 0.5 * hy,
                        0.5 * problem.physdims[1] - 0.5 * hy,
                    ],
                    [problem.physdims[2], problem.physdims[2] - well_hight * hz],
                ]
            ),
        ]
    )
    well_num_cells = 4

    problem.mdg = add_wells(
        problem.mdg, problem.domain, well_coords, well_num_cells, tol
    )

    # create the discretization
    discr = Flow(problem.mdg, discr=MVEMCodim2)

    # compute the effective speed-dependent permeability
    data.get_perm_factor(region=region)

    for sd, d in problem.mdg.subdomains(return_data=True):
        flux = np.zeros((3, sd.num_cells))
        pp.set_solution_values(Flow.P0_flux, flux, d, 0)
        pp.set_solution_values(Flow.P0_flux + "_old", flux.copy(), d, 0)

    variable_to_export += problem.save_perm()

    # non-linear problem solution with a fixed point strategy
    err_non_linear = max_err_non_linear + 1
    iteration_non_linear = 0
    while (
        err_non_linear > max_err_non_linear
        and iteration_non_linear < max_iteration_non_linear
    ):
        # solve the linearized problem
        discr.set_data(data.get())

        A, b = discr.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        discr.extract(x, u_bar=1)

        # compute the exit condition
        all_flux = np.empty((3, 0))
        all_flux_old = np.empty((3, 0))
        all_cell_volumes = np.empty(0)
        for sd, d in problem.mdg.subdomains(return_data=True):
            # collect the current flux
            flux = d[pp.TIME_STEP_SOLUTIONS][Flow.P0_flux][0]
            all_flux = np.hstack((all_flux, flux))
            # collect the old flux
            flux_old = d[pp.TIME_STEP_SOLUTIONS][Flow.P0_flux + "_old"][0]
            all_flux_old = np.hstack((all_flux_old, flux_old))
            # collect the cell volumes
            all_cell_volumes = np.hstack((all_cell_volumes, sd.cell_volumes))
            # save the old flux
            pp.set_solution_values(Flow.P0_flux + "_old", flux, d, 0)

        # compute the error and normalize the result
        err_non_linear = np.sum(
            all_cell_volumes * np.linalg.norm(all_flux - all_flux_old, axis=0)
        )
        norm_flux_old = np.sum(all_cell_volumes * np.linalg.norm(all_flux_old, axis=0))
        err_non_linear = (
            err_non_linear / norm_flux_old if norm_flux_old != 0 else err_non_linear
        )

        # exporter
        save = pp.Exporter(problem.mdg, "sol_" + file_name, folder_name=folder_name)
        save.write_vtu(variable_to_export, time_step=iteration_non_linear)

        print(
            "iteration non-linear problem",
            iteration_non_linear,
            "error",
            err_non_linear,
        )
        iteration_non_linear += 1

    save.write_pvd(np.arange(iteration_non_linear))
    write_network_pvd(file_name, folder_name, np.arange(iteration_non_linear))

    for sd, d in problem.mdg.subdomains(return_data=True):
        if region is None:
            np.savetxt(Flow.region, d[pp.TIME_STEP_SOLUTIONS][Flow.region][0])
        flux = d[pp.TIME_STEP_SOLUTIONS][Flow.P0_flux][0]
        pressure = d[pp.TIME_STEP_SOLUTIONS][Flow.pressure][0]

    return flux, pressure


# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    print("Perform the adaptative scheme")
    q_adapt, p_adapt = main(None)
    # print("Perform the heterogeneous scheme")
    # q_hete, p_hete = main("region")
    # print("Perform the darcy-based scheme")
    # q_darcy, p_darcy = main("examples/case4/region_darcy")
    # print("Perform the forshheimer-based scheme")
    # q_forsh, p_forsh = main("region_forsh")

    compute_errors = False
    if compute_errors:
        p = (p_darcy, p_forch, p_hete)
        q = (q_darcy, q_forch, q_hete)
        compute_error(mdg, p, q)
