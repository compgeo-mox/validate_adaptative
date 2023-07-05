import numpy as np
import porepy as pp
import scipy.sparse as sps

from problem import Problem
from parameters import Parameters
from data import Data

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
import tags
from exporter import write_network_pvd, make_file_name
import time

# ------------------------------------------------------------------------------#

def main(region, parameters, problem, data):
    # set files and folders to work with
    file_name = "case2"
    if region == None:
        folder_name = "./solutions/adaptive/"
    elif region == "region":
        folder_name = "./solutions/heterogeneous/"
    elif region == "region_darcy":
        folder_name = "./solutions/darcy/"
    elif region == "region_forch":
        folder_name = "./solutions/forch/"

    # variables to visualize
    variable_to_export = [Flow.pressure, \
                          Flow.P0_flux, \
                          Flow.permeability, \
                          Flow.P0_flux_norm, \
                          Flow.region]

    # parameters for nonlinear solver
    max_iteration_non_linear = 100
    max_err_non_linear = 1e-8

    # create the discretization
    discr = Flow(problem.mdg, discr = pp.MVEM)

    # compute the effective flux-dependent permeability
    data.get_perm_factor(region=region)

    for sd, d in problem.mdg.subdomains(return_data=True):
        flux = np.zeros((3, sd.num_cells))
        pp.set_solution_values(Flow.P0_flux, flux, d, 0)
        pp.set_solution_values(Flow.P0_flux + "_old", flux.copy(), d, 0)

    variable_to_export += problem.save_perm() # add intrinsic permeability to visualize
    variable_to_export += problem.save_forch_vars() # add Forchheimer number and other vars

    # non-linear problem solution with a fixed point strategy
    err_non_linear = max_err_non_linear + 1
    iteration_non_linear = 0
    while err_non_linear > max_err_non_linear and iteration_non_linear < max_iteration_non_linear:

        # solve the linearized problem
        discr.set_data(data.get())

        A, b = discr.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        discr.extract(x, u_bar=1) # u_bar=1 here because fluxes are normalized by u_bar

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
        err_non_linear = np.sum(all_cell_volumes * np.linalg.norm(all_flux - all_flux_old, axis=0))
        norm_flux_old = np.sum(all_cell_volumes * np.linalg.norm(all_flux_old, axis=0))
        err_non_linear = err_non_linear / norm_flux_old if norm_flux_old != 0 else err_non_linear

        # save new Forchheimer number
        problem.save_forch_vars(flux=all_flux)

        # exporter
        save = pp.Exporter(problem.mdg, "sol_" + file_name, folder_name=folder_name)
        save.write_vtu(variable_to_export, time_step=iteration_non_linear)

        print("iteration non-linear problem", iteration_non_linear, "error", err_non_linear)
        iteration_non_linear += 1

    save.write_pvd(np.arange(iteration_non_linear))
    write_network_pvd(file_name, folder_name, np.arange(iteration_non_linear))

    for sd, d in problem.mdg.subdomains(return_data=True):
        if region is None:
            np.savetxt("./regions/" + Flow.region, d[pp.TIME_STEP_SOLUTIONS][Flow.region][0])
        flux = d[pp.TIME_STEP_SOLUTIONS][Flow.P0_flux][0]
        pressure = d[pp.TIME_STEP_SOLUTIONS][Flow.pressure][0]

    return flux, pressure, problem.mdg

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    parameters = Parameters()        # get parameters, print them and read porosity
    problem = Problem(parameters)    # create the grid bucket and get intrinsic permeability
    data = Data(parameters, problem) # get data for computation of flux-dependent permeability

    # run the various schemes
    print("", "---- Perform the adaptive scheme ----", sep="\n")
    start = time.time()
    q_adapt, p_adapt, mdg = main(None, parameters, problem, data)
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the heterogeneous scheme ----", sep="\n")
    start = time.time()
    q_hete, p_hete, _ = main("region", parameters, problem, data)
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the Darcy scheme ----", sep="\n")
    start = time.time()
    q_darcy, p_darcy, _ = main("region_darcy", parameters, problem, data)
    end = time.time()
    print("run time =", end - start, "[s]")

    print("", "---- Perform the Forchheimer scheme ----", sep="\n")
    start = time.time()
    q_forch, p_forch, _ = main("region_forch", parameters, problem, data)
    end = time.time()
    print("run time =", end - start, "[s]")

    # compute the errors with respect to reference scheme (Forchheimer)
    compute_errors = True;

    if compute_errors:
        print(" ", "---- Compute region-wise errors with respect to the Forchheimer scheme ----", \
              sep="\n")
        region = np.loadtxt("./regions/region").astype(bool)
        p_ref = p_forch
        q_ref = q_forch

        for reg in np.unique(region):
            pos = region == reg
            # mass matrix
            mass = sps.diags([sd.cell_volumes[pos] for sd in mdg.subdomains()], [0])

            norm_scalar = lambda x: np.sqrt(x @ mass @ x)
            norm_vector = lambda x: np.sqrt(np.linalg.norm(x, axis=0) @ mass @ np.linalg.norm(x, axis=0))

            # we assume the Forchheimer solution to be the reference
            norm_p_ref = norm_scalar(p_ref[pos])
            norm_q_ref = norm_vector(q_ref[:, pos])

            # let's compute the errors
            err_p_hete = norm_scalar(p_ref[pos] - p_hete[pos]) / norm_p_ref
            err_q_hete = norm_vector(q_ref[:, pos] - q_hete[:, pos]) / norm_q_ref

            err_p_darcy = norm_scalar(p_ref[pos] - p_darcy[pos]) / norm_p_ref
            err_q_darcy = norm_vector(q_ref[:, pos] - q_darcy[:, pos]) / norm_q_ref

            err_p_forch = norm_scalar(p_ref[pos] - p_forch[pos]) / norm_p_ref
            err_q_forch = norm_vector(q_ref[:, pos] - q_forch[:, pos]) / norm_q_ref

            print("------")
            print("In", {True: "Darcy", False: "Forchheimer"}[reg], "region")
            print("------")

            print("Errors for the heterogeneous scheme:")
            print("Pressure", err_p_hete)
            print("Flux", err_q_hete)
            print("------")

            print("Errors for the Darcy scheme:")
            print("Pressure", err_p_darcy)
            print("Flux", err_q_darcy)
            print("------")

            print("Errors for the Forchheimer scheme:")
            print("Pressure", err_p_forch)
            print("Flux", err_q_forch)

            
        print(" ", "---- Compute global errors with respect to the Forchheimer scheme ----", \
              sep="\n")

        # mass matrix
        mass = sps.diags([sd.cell_volumes for sd in mdg.subdomains()], [0])

        norm_scalar = lambda x: np.sqrt(x @ mass @ x)
        norm_vector = lambda x: np.sqrt(np.linalg.norm(x, axis=0) @ mass @ np.linalg.norm(x, axis=0))

        # we assume the Forchheimer solution to be the reference
        norm_p_ref = norm_scalar(p_ref)
        norm_q_ref = norm_vector(q_ref)

        # let's compute the errors
        err_p_hete = norm_scalar(p_ref - p_hete) / norm_p_ref
        err_q_hete = norm_vector(q_ref - q_hete) / norm_q_ref

        err_p_darcy = norm_scalar(p_ref - p_darcy) / norm_p_ref
        err_q_darcy = norm_vector(q_ref - q_darcy) / norm_q_ref

        err_p_forch = norm_scalar(p_ref - p_forch) / norm_p_ref
        err_q_forch = norm_vector(q_ref - q_forch) / norm_q_ref

        print("------")
        print("In whole domain")
        print("------")

        print("Errors for the heterogeneous scheme:")
        print("Pressure", err_p_hete)
        print("Flux", err_q_hete)
        print("------")

        print("Errors for the Darcy scheme:")
        print("Pressure", err_p_darcy)
        print("Flux", err_q_darcy)
        print("------")

        print("Errors for the Forchheimer scheme:")
        print("Pressure", err_p_forch)
        print("Flux", err_q_forch)
