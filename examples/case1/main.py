import numpy as np
import porepy as pp
import time
import scipy.sparse as sps

from problem import Problem
from parameters import Parameters
from data import Data

import sys

sys.path.insert(0, "./src/")
from flow import Flow

# import tags
from exporter import write_network_pvd, make_file_name
from compute_error import compute_error

# ------------------------------------------------------------------------------#

main_folder = "examples/case1/"


def main(region, parameters, problem, data, out_folder):
    # set files and folders to work with
    file_name = "case1"
    if region == None:
        folder_name = out_folder + "solutions/adaptive/"
    elif region == "region":
        folder_name = out_folder + "solutions/heterogeneous/"
    elif region == "region_darcy":
        folder_name = out_folder + "solutions/darcy/"
    elif region == "region_forch":
        folder_name = out_folder + "solutions/forch/"

    # variables to visualize
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

    # create the discretization
    discr = Flow(problem.mdg, discr=pp.MVEM)

    # compute the effective flux-dependent permeability
    data.get_perm_factor(region=region)

    for sd, d in problem.mdg.subdomains(return_data=True):
        flux = np.zeros((3, sd.num_cells))
        pp.set_solution_values(Flow.P0_flux, flux, d, 0)
        pp.set_solution_values(Flow.P0_flux + "_old", flux.copy(), d, 0)

    variable_to_export += problem.save_perm()  # add intrinsic permeability to visualize
    variable_to_export += (
        problem.save_forch_vars()
    )  # add Forchheimer number and other vars

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
        problem.save_forch_vars(flux=all_flux)

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
        if region is None:
            file_name_region = out_folder + "regions/" + Flow.region
            np.savetxt(file_name_region, d[pp.TIME_STEP_SOLUTIONS][Flow.region][0])
        flux = d[pp.TIME_STEP_SOLUTIONS][Flow.P0_flux][0]
        pressure = d[pp.TIME_STEP_SOLUTIONS][Flow.pressure][0]

    return flux, pressure, problem.mdg


# ------------------------------------------------------------------------------#


def run_test(layer, val_well, main_folder, out_folder, E):
    parameters = Parameters(
        main_folder, val_well=val_well, layers=layer, E=E
    )  # get parameters, print them and read porosity
    problem = Problem(
        parameters
    )  # create the grid bucket and get intrinsic permeability
    data = Data(
        parameters, problem, out_folder
    )  # get data for computation of flux-dependent permeability

    # run the various schemes
    print("", "---- Perform the adaptive scheme ----", sep="\n")
    start = time.time()
    q_adapt, p_adapt, mdg = main(None, parameters, problem, data, out_folder)
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the heterogeneous scheme ----", sep="\n")
    start = time.time()
    q_hete, p_hete, _ = main("region", parameters, problem, data, out_folder)
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the Darcy scheme ----", sep="\n")
    start = time.time()
    q_darcy, p_darcy, _ = main("region_darcy", parameters, problem, data, out_folder)
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the Forchheimer scheme ----", sep="\n")
    start = time.time()
    q_forch, p_forch, _ = main("region_forch", parameters, problem, data, out_folder)
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
            10,
            20,
            40,
            100,
            200,
            400,
            1000,
            2000,
            4000,
            10000,
            20000,
            40000,
            100000,
            200000,
            400000,
        ]
    )

    layers = [35]
    vals_well = [50, 200]  # 10 50 200

    for v in vals_well:
        for l in layers:
            err_p = []
            err_q = []
            num_F = []
            for inv_E in inv_E_vec:
                folder = "case_l_" + str(l) + "_v_" + str(v) + "/"
                out = run_test(l, v, main_folder, main_folder + folder, 1 / inv_E)

                err_p.append(out[0])
                err_q.append(out[1])
                num_F.append((out[2].size - out[2].sum()) / out[2].size)

            print(err_p, err_q, num_F)

            file_name = main_folder + "output" + str(l) + "_" + str(v) + ".txt"
            np.savetxt(file_name, (inv_E_vec, err_p, err_q, num_F), delimiter=",")
