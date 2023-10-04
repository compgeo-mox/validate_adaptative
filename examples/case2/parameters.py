import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


class Parameters:
    def __init__(self, folder, subcase="couple"):
        # 2d box geometry
        self.length_x = 10  # box x-axis size [m]
        self.length_y = 10  # box y-axis size [m]

        # physical parameters
        self.mu = 0.001  # fluid's dynamic viscosity [Pa.s]
        self.rho = 998.0  # fluid's density [kg/m3]
        self.c_F = 0.55  # Forchheimer coefficient [-]
        self.g = 9.81  # gravity [m/s2]

        # boundary parameters
        self.influx = 0.007  # fluid's influx at top boundary [kg/m2/s]
        self.atm_pressure = 1.01325e5  # atmospheric pressure [Pa]

        # high-permeability lenses
        self.data_kind = "poro"  # kind = "poro" or "perm"

        if subcase == "couple":
            lens_1 = dict(bounds=lambda x, y: 3 < x < 3.2 and 0 < y < 10, val=0.94)
            lens_2 = dict(bounds=lambda x, y: 7.2 < x < 7.6 and 0 < y < 10, val=0.94)
            self.lenses = [lens_1, lens_2]
        elif subcase == "network":
            lens_v1 = dict(bounds=lambda x, y: 4 < x < 4.2 and 0.5 < y < 9, val=0.99)
            lens_v2 = dict(bounds=lambda x, y: 7.2 < x < 7.4 and 3 < y < 8.8, val=0.94)
            lens_v3 = dict(
                bounds=lambda x, y: 1.2 < x < 1.5 and 0.7 < y < 9.6, val=0.97
            )
            lens_h1 = dict(bounds=lambda x, y: 0.6 < x < 7 and 1 < y < 1.2, val=0.91)
            lens_h2 = dict(bounds=lambda x, y: 3.2 < x < 9 and 5 < y < 5.2, val=0.98)
            lens_h3 = dict(bounds=lambda x, y: 0.9 < x < 8.5 and 6 < y < 6.2, val=0.95)
            lens_h4 = dict(bounds=lambda x, y: 0.6 < x < 8 and 8.4 < y < 8.7, val=0.92)
            self.lenses = [
                lens_v1,
                lens_v2,
                lens_v3,
                lens_h1,
                lens_h2,
                lens_h3,
                lens_h4,
            ]

        # to determine critical Forchheimer number
        self.E = 0.1  # maximum error to Forchheimer accepted [-]

        # call internal functions
        file_bg = (
            folder + "porosity"
        )  # file containing background porosity or permeability
        self._read_background(file_bg)
        self._compute_and_print()

    # ------------------------------------------------------------------------------#

    def _read_background(self, file_bg):
        assert (
            self.data_kind == "poro" or self.data_kind == "perm"
        ), "Background data provided must be of porosity or permeability type"

        # the lines below may be changed depending on the format of the background data file #
        lines_bg = open(file_bg, "r").read().split("\n")

        lines_bg_list = []
        for line in lines_bg[1:]:
            line_float = [float(line.split(" ")[j]) for j in range(3)]
            lines_bg_list = lines_bg_list + [line_float]

        bg_list = []
        if self.data_kind == "poro":
            for line in lines_bg_list:
                bg_list = bg_list + [0.35 * np.power(10, line[2])]
        else:
            for line in lines_bg_list:
                bg_list = bg_list + [lines[2]]
        ##

        self.bg_array = np.asarray(bg_list)  # array of bg porosity of permeability

        lines_bg_array = np.asarray(lines_bg_list)
        self.num_cells_x = int(
            np.max(lines_bg_array[:, 0])
        )  # number of cells on x-axis
        self.num_cells_y = int(
            np.max(lines_bg_array[:, 1])
        )  # number of cells on y-axis

    # ------------------------------------------------------------------------------#

    def _compute_and_print(self):
        # compute parameters from given ones above
        self.nu = self.mu / self.rho  # fluid's kinematic viscosity [m2/s]
        self.Fo_c = self.E / (1 - self.E)  # critical Forchheimer number [-]
        self.beta = self.c_F / self.rho  # Forchheimer drag coefficient [m3/kg]

        # print newly computed parameters
        print("---- Print the parameters ----")
        print(
            "mu =",
            round(self.mu, 5),
            "[Pa.s]",
            " rho =",
            round(self.rho, 5),
            "[kg/m3]",
            " influx =",
            round(self.influx, 5),
            "[kg/m2/s]",
            " Fo_c =",
            round(self.Fo_c, 5),
            "[-]",
        )

        num_lenses = len(self.lenses)
        if self.data_kind == "poro":
            for l in range(num_lenses):
                print(
                    "phi_" + str(l + 1) + " =",
                    round(self.lenses[l]["val"] * 100, 5),
                    "[%]",
                )
        else:
            for l in range(num_lenses):
                print("K_" + str(l + 1) + " =", round(self.lenses[l]["val"], 5), "[m2]")
