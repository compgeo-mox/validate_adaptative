import numpy as np
import porepy as pp
import scipy.sparse as sps

from data import Data
from problem import Problem

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
import tags
from exporter import write_network_pvd, make_file_name

# ------------------------------------------------------------------------------#

def main(region):

    # tolerance in the computation
    tol = 1e-10

    # assign the flag for the low permeable fractures
    epsilon = 1e-2

    file_name = "case2"
    if region == None:
        folder_name = "./sol/adaptative/"
    elif region == "region":
        folder_name = "./sol/heterogeneous/"
    elif region == "region_darcy":
        folder_name = "./sol/darcy/"
    elif region == "region_forsh":
        folder_name = "./sol/forsh/"

    variable_to_export = [Flow.pressure, Flow.P0_flux, Flow.permeability, Flow.P0_flux_norm, Flow.region, Flow.gradient_pressure]

    max_iteration_non_linear = 40
    max_err_non_linear = 1e-8

    # create the grid bucket
    problem = Problem()
    problem.read_perm()

    # create the discretization
    discr = Flow(problem.mdg, discr = pp.MVEM)
    test_data = Data(problem, epsilon=epsilon, region=region)

    for sd, d in problem.mdg.subdomains(return_data=True):
        d.update({pp.STATE: {}})
        flux = np.zeros((3, sd.num_cells))
        d[pp.STATE].update({Flow.P0_flux: flux})
        d[pp.STATE].update({Flow.P0_flux + "_old": flux})

    variable_to_export += problem.save_perm()

    # non-linear problem solution with a fixed point strategy
    err_non_linear = max_err_non_linear + 1
    iteration_non_linear = 0
    while err_non_linear > max_err_non_linear and iteration_non_linear < max_iteration_non_linear:

        # solve the linearized problem
        discr.set_data(test_data.get())

        A, b = discr.matrix_rhs()
        x = sps.linalg.spsolve(A, b)
        discr.extract(x, u_bar=1)

        # compute the exit condition
        all_flux = np.empty((3, 0))
        all_flux_old = np.empty((3, 0))
        all_cell_volumes = np.empty(0)
        for sd, d in problem.mdg.subdomains(return_data=True):
            # collect the current flux
            flux = d[pp.STATE][Flow.P0_flux]
            all_flux = np.hstack((all_flux, flux))
            # collect the old flux
            flux_old = d[pp.STATE][Flow.P0_flux + "_old"]
            all_flux_old = np.hstack((all_flux_old, flux_old))
            # collect the cell volumes
            all_cell_volumes = np.hstack((all_cell_volumes, sd.cell_volumes))
            # save the old flux
            d[pp.STATE][Flow.P0_flux + "_old"] = flux

        # compute the error and normalize the result
        err_non_linear = np.sum(all_cell_volumes * np.linalg.norm(all_flux - all_flux_old, axis=0))
        norm_flux_old = np.sum(all_cell_volumes * np.linalg.norm(all_flux_old, axis=0))
        err_non_linear = err_non_linear / norm_flux_old if norm_flux_old != 0 else err_non_linear

        # exporter
        save = pp.Exporter(problem.mdg, "sol_" + file_name, folder_name=folder_name)
        save.write_vtu(variable_to_export, time_step=iteration_non_linear)

        print("iteration non-linear problem", iteration_non_linear, "error", err_non_linear)
        iteration_non_linear += 1

    save.write_pvd(np.arange(iteration_non_linear))
    write_network_pvd(file_name, folder_name, np.arange(iteration_non_linear))

    for sd, d in problem.mdg.subdomains(return_data=True):
        if region is None:
            np.savetxt(Flow.region, d[pp.STATE][Flow.region])
        flux = d[pp.STATE][Flow.P0_flux]
        pressure = d[pp.STATE][Flow.pressure]

    return flux, pressure, problem.mdg

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    print("Perform the adaptative scheme")
    q_adapt, p_adapt, mdg = main(None)
    print("Perform the heterogeneous scheme")
    q_hete, p_hete, _ = main("region")
    print("Perform the darcy-based scheme")
    q_darcy, p_darcy, _ = main("region_darcy")
    print("Perform the forshheimer-based scheme")
    q_forsh, p_forsh, _ = main("region_forsh")

    region = np.loadtxt("region").astype(bool)
    p_ref = p_forsh
    q_ref = q_forsh

    for reg in np.unique(region):
        pos = region == reg
        # mass matrix
        mass = sps.diags([sd.cell_volumes[pos] for sd in mdg.subdomains()], [0])

        norm_scalar = lambda x: np.sqrt(x @ mass @ x)
        norm_vector = lambda x: np.sqrt(np.linalg.norm(x, axis=0) @ mass @ np.linalg.norm(x, axis=0))

        # we assume to be the adaptative solution as the reference
        norm_p_ref = norm_scalar(p_ref[pos])
        norm_q_ref = norm_vector(q_ref[:, pos])

        # let's compute the errors
        err_p_hete = norm_scalar(p_ref[pos] - p_hete[pos]) / norm_p_ref
        err_q_hete = norm_vector(q_ref[:, pos] - q_hete[:, pos]) / norm_q_ref

        err_p_darcy = norm_scalar(p_ref[pos] - p_darcy[pos]) / norm_p_ref
        err_q_darcy = norm_vector(q_ref[:, pos] - q_darcy[:, pos]) / norm_q_ref

        err_p_forsh = norm_scalar(p_ref[pos] - p_forsh[pos]) / norm_p_ref
        err_q_forsh = norm_vector(q_ref[:, pos] - q_forsh[:, pos]) / norm_q_ref

        print("Region", {True: "darcy", False: "forsh"}[reg])
        print("------")

        print("Errors for case hete:")
        print("pressure", err_p_hete)
        print("flux", err_q_hete)
        print("------")

        print("Errors for case darcy:")
        print("pressure", err_p_darcy)
        print("flux", err_q_darcy)
        print("------")

        print("Errors for case forsh:")
        print("pressure", err_p_forsh)
        print("flux", err_q_forsh)
        print("------")

    import pdb; pdb.set_trace()
