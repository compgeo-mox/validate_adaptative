import numpy as np
import porepy as pp
import time
import scipy.sparse as sps

from data import Data
import sys

sys.path.insert(0, "./examples/case1/")
from problem import Problem
from parameters import Parameters

import sys

sys.path.insert(0, "./src/")
from flow import Flow
from exporter import write_network_pvd
from solver_codim2 import MVEMCodim2
from compute_error import compute_error

# ------------------------------------------------------------------------------#

main_folder = "./examples/case4/"
main_folder_case1 = "./examples/case1/"


def add_wells(mdg, domain, well_coords, well_num_cells, tol):
    # define the wells
    wells = np.array([pp.Well(e) for e in well_coords])

    # the mesh size is determined by the length of the wells and the imposed num_cells
    wells_length = np.array([np.linalg.norm(e[:, 1] - e[:, 0]) for e in well_coords])
    mesh_size = np.amin(wells_length / well_num_cells)

    # create the wells grids
    wells_set = pp.WellNetwork3d(
        domain, wells, parameters={"mesh_size": mesh_size}, tol=tol
    )

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


def main(region, parameters, problem, data, out_folder, well_height):
    # tolerance in the computation
    tol = 1e-4

    # set files and folders to work with
    file_name = "case4"
    if region == None:
        folder_name = out_folder + "solutions/adaptive/"
    elif region == "region":
        folder_name = out_folder + "solutions/heterogeneous/"
    elif region == "region_darcy":
        folder_name = out_folder + "solutions/darcy/"
        file_name_region = out_folder + "regions/" + region
        np.savetxt(file_name_region, np.ones(problem.sd.num_cells))
    elif region == "region_forch":
        folder_name = out_folder + "solutions/forch/"
        file_name_region = out_folder + "regions/" + region
        np.savetxt(file_name_region, np.zeros(problem.sd.num_cells))

    variable_to_export = [
        Flow.pressure,
        Flow.P0_flux,
        Flow.permeability,
        Flow.P0_flux_norm,
        Flow.region,
    ]

    # parameters for nonlinear solver
    max_iteration_non_linear = 100
    max_err_non_linear = 1e-8

    hx, hy, hz = problem.physdims / problem.shape

    # add the wells
    well_coords = np.asarray(
        [
            np.array(
                [
                    [0.5 * hx, 0.5 * hx],
                    [0.5 * hy, 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_height * hz],
                ]
            ),
            np.array(
                [
                    [problem.physdims[0] - 0.5 * hx, problem.physdims[0] - 0.5 * hx],
                    [0.5 * hy, 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_height * hz],
                ]
            ),
            np.array(
                [
                    [0.5 * hx, 0.5 * hx],
                    [problem.physdims[1] - 0.5 * hy, problem.physdims[1] - 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_height * hz],
                ]
            ),
            np.array(
                [
                    [problem.physdims[0] - 0.5 * hx, problem.physdims[0] - 0.5 * hx],
                    [problem.physdims[1] - 0.5 * hy, problem.physdims[1] - 0.5 * hy],
                    [problem.physdims[2], problem.physdims[2] - well_height * hz],
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
                    [problem.physdims[2], problem.physdims[2] - well_height * hz],
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

    variable_to_export += problem.save_perm()  # add intrinsic permeability to visualize
    # variable_to_export += problem.save_forch_vars()  # add Forchheimer number and other vars

    # non-linear problem solution with a fixed point strategy
    err_non_linear = max_err_non_linear + 1
    iteration_non_linear = 0
    while (
        err_non_linear > max_err_non_linear
        and iteration_non_linear < max_iteration_non_linear
    ):
        # solve the linearized problem
        discr.set_data(data.get())

        print("Assemble the linearized problem")
        A, b = discr.matrix_rhs()

        print("Solve the linearized problem", A.shape, b.shape)
        if parameters.bdry_conditions == "dir":
            x = sps.linalg.spsolve(A, b)
        else:
            vect = np.zeros((1, A.shape[0]))
            num_cells = problem.mdg.num_subdomain_cells()
            vect[0, -num_cells:] = 1
            A = sps.bmat([[A, vect.T], [vect, None]], format="csc")
            b = np.concatenate((b, [0]))
            x = sps.linalg.spsolve(A, b)[:-1]

        discr.extract(x, u_bar=1)  # u_bar=1 here because fluxes are normalized by u_bar

        print("Compute the exit condition")
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

        # save new Forchheimer number
        # problem.save_forch_vars(flux=all_flux)

        print("Export the solution")
        # exporter
        save = pp.Exporter(problem.mdg, "sol_" + file_name, folder_name=folder_name)
        save.write_vtu(variable_to_export, time_step=iteration_non_linear)

        print(
            "iteration non-linear problem",
            iteration_non_linear,
            "error",
            err_non_linear,
            "file_name",
            folder_name + "/sol_" + file_name,
        )
        iteration_non_linear += 1

    save.write_pvd(np.arange(iteration_non_linear))
    write_network_pvd(file_name, folder_name, np.arange(iteration_non_linear))

    for sd, d in problem.mdg.subdomains(return_data=True):
        if sd.dim == 3:
            if region is None:
                file_name_region = out_folder + "regions/" + Flow.region
                np.savetxt(file_name_region, d[pp.TIME_STEP_SOLUTIONS][Flow.region][0])
            flux = d[pp.TIME_STEP_SOLUTIONS][Flow.P0_flux][0]
            pressure = d[pp.TIME_STEP_SOLUTIONS][Flow.pressure][0]

    return flux, pressure, problem.mdg


# ------------------------------------------------------------------------------#


def run_test(layers, val_well, main_folder, out_folder, E, well_height):
    parameters = Parameters(
        main_folder_case1, val_well=val_well, layers=layers, E=E  # 35
    )  # get parameters, print them and read porosity
    problem = Problem(
        parameters,
        pos_x=np.arange(10),  # + 15,
        pos_y=np.arange(20),  # + 40  # 20 20
    )  # create the grid bucket and get intrinsic permeability
    data = Data(
        parameters, problem, out_folder
    )  # get data for computation of speed-dependent permeability

    # run the various schemes
    print("", "---- Perform the adaptive scheme ----", sep="\n")
    start = time.time()
    q_adapt, p_adapt, mdg = main(
        None, parameters, problem, data, out_folder, well_height
    )
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the heterogeneous scheme ----", sep="\n")
    start = time.time()
    q_hete, p_hete, _ = main(
        "region", parameters, problem, data, out_folder, well_height
    )
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the Darcy scheme ----", sep="\n")
    start = time.time()
    q_darcy, p_darcy, _ = main(
        "region_darcy", parameters, problem, data, out_folder, well_height
    )
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the Forchheimer scheme ----", sep="\n")
    start = time.time()
    q_forch, p_forch, _ = main(
        "region_forch", parameters, problem, data, out_folder, well_height
    )
    end = time.time()
    print("run time =", end - start, "[s]")

    # compute the errors with respect to reference scheme (Forchheimer)
    compute_errors = True

    if compute_errors:
        p = (p_darcy, p_forch, p_hete)
        q = (q_darcy, q_forch, q_hete)
        err_p_hete, err_q_hete, _, _, _, _, region = compute_error(
            mdg, *p, *q, folder=out_folder
        )

    return err_p_hete, err_q_hete, region


if __name__ == "__main__":
    inv_E_vec = np.array(
        [
            4,
            20,
            100,
            400,
            2000,
            10000,
            40000,
            200000,
        ]
    )

    layers = np.arange(15)
    vals_well = [10]
    well_height = [3.5, 7.5, 9.5]

    for v in vals_well:
        for w in well_height:
            err_p = []
            err_q = []
            num_F = []
            for inv_E in inv_E_vec:
                folder = "case_w_" + str(w) + "_v_" + str(v) + "/"
                out = run_test(
                    layers, v, main_folder, main_folder + folder, 1 / inv_E, w
                )

                err_p.append(out[0])
                err_q.append(out[1])
                num_F.append((out[2].size - out[2].sum()) / out[2].size)

            print(err_p, err_q, num_F)

            file_name = main_folder + "output_w_" + str(w) + "_" + str(v) + ".txt"
            np.savetxt(file_name, (inv_E_vec, err_p, err_q, num_F), delimiter=",")
