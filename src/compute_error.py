import numpy as np
import porepy as pp
import scipy.sparse as sps


def compute_error(
    mdg,
    p_darcy,
    p_forch,
    p_hete,
    q_darcy,
    q_forch,
    q_hete,
    folder="./",
    file_region="regions/region",
):
    print(
        " ",
        "---- Compute region-wise errors with respect to the Forchheimer scheme ----",
        sep="\n",
    )
    region = np.loadtxt(folder + file_region).astype(bool)
    p_ref = p_forch
    q_ref = q_forch

    for reg in np.unique(region):
        pos = region == reg
        # mass matrix
        vols = []
        for sd in mdg.subdomains():
            if sd.dim != 1 or sd.well_num <= -1:
                vols.append(sd.cell_volumes[pos])
        mass = sps.diags(vols, [0])
        #mass = sps.diags([sd.cell_volumes[pos] for sd in mdg.subdomains()], [0])

        norm_scalar = lambda x: np.sqrt(x @ mass @ x)
        norm_vector = lambda x: np.sqrt(
            np.linalg.norm(x, axis=0) @ mass @ np.linalg.norm(x, axis=0)
        )

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

    print(
        " ",
        "---- Compute global errors with respect to the Forchheimer scheme ----",
        sep="\n",
    )

    pos = [True for i in range(len(region))]
    # mass matrix
    vols = []
    for sd in mdg.subdomains():
        if sd.dim != 1 or sd.well_num <= -1:
            vols.append(sd.cell_volumes[pos])
    mass = sps.diags(vols, [0])
    #mass = sps.diags([sd.cell_volumes for sd in mdg.subdomains()], [0])

    norm_scalar = lambda x: np.sqrt(x @ mass @ x)
    norm_vector = lambda x: np.sqrt(
        np.linalg.norm(x, axis=0) @ mass @ np.linalg.norm(x, axis=0)
    )

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
