import numpy as np
import porepy as pp
import scipy.sparse as sps

from data import Data
from spe10 import Spe10

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
import tags
from exporter import write_network_pvd, make_file_name

# ------------------------------------------------------------------------------#

def main():

    # tolerance in the computation
    tol = 1e-10

    # assign the flag for the low permeable fractures
    epsilon = 1e-1
    u_bar = 1e-7 # 1 0.5 0.25 0.125 0.0625 0.03125

    file_name = "case1"
    folder_name = "./adaptative/"
    variable_to_export = [Flow.pressure, Flow.P0_flux, Flow.permeability, Flow.P0_flux_norm, Flow.region]

    max_iteration_non_linear = 20
    max_err_non_linear = 1e-4

    # create the grid bucket
    spe10 = Spe10(35)
    spe10.read_perm("./spe10_perm/")

    # create the discretization
    discr = Flow(spe10.mdg, discr = pp.MVEM)
    test_data = Data(spe10, epsilon, u_bar)

    for sd, d in spe10.mdg.subdomains(return_data=True):
        d.update({pp.STATE: {}})
        flux = np.zeros((3, sd.num_cells))
        d[pp.STATE].update({Flow.P0_flux: flux})
        d[pp.STATE].update({Flow.P0_flux + "_old": flux})

    variable_to_export += spe10.save_perm()

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
        for sd, d in spe10.mdg.subdomains(return_data=True):
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
        save = pp.Exporter(spe10.mdg, "sol_" + file_name, folder_name=folder_name)
        save.write_vtu(variable_to_export, time_step=iteration_non_linear)

        print("iteration non-linear problem", iteration_non_linear, "error", err_non_linear)
        iteration_non_linear += 1

    save.write_pvd(np.arange(iteration_non_linear))
    write_network_pvd(file_name, folder_name, np.arange(iteration_non_linear))

    for sd, d in spe10.mdg.subdomains(return_data=True):
        np.savetxt(Flow.region, d[pp.STATE][Flow.region])

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
