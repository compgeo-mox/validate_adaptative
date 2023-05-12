import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

class Parameters:

    def __init__(self):
        # layer
        self.layer = 35               # layer of SPE10 case to be considered
        self.layer_depth = 21.336     # depth [m] of above layer

        # physical parameters
        self.mu = 0.0003 # fluid's viscosity [Pa.s]
        self.rho = 1025. # fluid's density [kg/m3]
        self.c_F = 0.55 # Forchheimer coefficient [-]
        self.g = 9.81   # gravity [m/s2]

        # boundary parameters
        self.atm_pressure = 1.01325e5  # atmospheric pressure [Pa]

        # wells
        well_i1 = dict(cell_id = 6629, val = 50) # well mass source [m3/s]
        well_p1 = dict(cell_id = 13140, val = -50)
        well_p2 = dict(cell_id = 13199, val = -50)
        well_p3 = dict(cell_id = 0, val = -50)
        well_p4 = dict(cell_id = 59, val = -50)
        self.wells = [well_i1, well_p1, well_p2, well_p3, well_p4]

        # to determine critical Forchheimer number
        self.E = 0.1    # maximum error to Forchheimer accepted [-]

        # call internal functions
        file_perm = "./spe10_perm/" + str(self.layer) + ".tar.gz" # background permeability file
        self._read_background(file_perm)
        self._compute_and_print()

    # ------------------------------------------------------------------------------#

    def _read_background(self, file_perm):
        perm_layer = np.loadtxt(file_perm, delimiter=",")
        self.perm_layer = perm_layer *pp.DARCY # multiply by pp.DARCY to change unit to [m2]

    # ------------------------------------------------------------------------------#

    def _compute_and_print(self):
        # compute parameters from given ones above
        self.Fo_c = self.E/(1-self.E)                     # critical Forchheimer number [-]
        self.beta = self.c_F*self.rho                     # Forchheimer drag coefficient [kg/m3]

        # print newly computed parameters
        print("---- Print the parameters ----")
        print("mu =", round(self.mu, 5), "[Pa.s]", \
              " rho =", round(self.rho, 5), "[kg/m3]", \
              " Fo_c =", round(self.Fo_c, 5), "[-]")

        num_wells = len(self.wells)
        for l in range(num_wells):
            print("q_well_" + str(l+1) + " =", round(self.wells[l]["val"], 5), "[m3/s]")
