import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


class Parameters:
    def __init__(self, folder, val_well=50, layers=35):
        # layer
        self.layers = np.sort(np.atleast_1d(layers))  # layer of SPE10 case to be considered
        self.layer_depth = 21.336  # depth [m] of above layer

        # physical parameters
        self.mu = 0.0003  # fluid's dynamic viscosity [Pa.s]
        self.rho = 1025.0  # fluid's density [kg/m3]
        self.c_F = 0.55  # Forchheimer coefficient [-]
        self.g = 9.81  # gravity [m/s2]
        self.m = 2 # nonlinearity exponent [-] (>= 2, 2 is Darcy-Forchheimer)

        # boundary parameters
        self.bdry_conditions = "neu"   # "neu" or "dir"
        self.atm_pressure = 1.01325e5  # atmospheric pressure [Pa]

        # wells
        well_i1 = dict(cell_id=6629, val=val_well)  # well mass source [kg/s]
        well_p1 = dict(cell_id=13140, val=-val_well / 4)
        well_p2 = dict(cell_id=13199, val=-val_well / 4)
        well_p3 = dict(cell_id=0, val=-val_well / 4)
        well_p4 = dict(cell_id=59, val=-val_well / 4)
        self.wells = [well_i1, well_p1, well_p2, well_p3, well_p4]

        # to determine critical Forchheimer number
        self.E = 0.1  # maximum error to Forchheimer accepted [-]

        # call internal functions
        self._read_background(folder)
        self._compute_and_print()

    # ------------------------------------------------------------------------------#

    def _read_background(self, folder):
        self.perm_layer = []
        for pos, layer in enumerate(self.layers):
            perm_file = folder + "spe10_perm/" + str(layer) + ".tar.gz"
            self.perm_layer.append(np.loadtxt(perm_file, delimiter=",") * pp.DARCY)

    # ------------------------------------------------------------------------------#

    def _compute_and_print(self):
        # compute parameters from given ones above
        self.nu = self.mu / self.rho  # fluid's kinematic viscosity [m2/s]
        self.Fo_c = self.E / (1 - self.E)  # critical Forchheimer number [-]

        # print newly computed parameters
        print("---- Print the parameters ----")
        print("mu =", round(self.mu, 5), "[Pa.s]",
              " rho =", round(self.rho, 5), "[kg/m3]",
              " Fo_c =", round(self.Fo_c, 5), "[-]")

        num_wells = len(self.wells)
        for l in range(num_wells):
            print(
                "q_well_" + str(l + 1) + " =", round(self.wells[l]["val"], 5), "[kg/s]"
            )
