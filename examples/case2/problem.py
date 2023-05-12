import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#

class Problem(object):

    def __init__(self, parameters, layers=1):
        self.parameters = parameters

        self.full_shape = (self.parameters.num_cells_x, \
                           self.parameters.num_cells_y, \
                           0)
        self.full_physdims = (self.parameters.length_x, \
                              self.parameters.length_y, \
                              0)

        self.layers = np.sort(np.atleast_1d(layers))

        self.N = 0
        self.n = 0
        self._compute_size()

        self.mdg = None
        self._create_mdg()

        self.perm = None
        self.layers_id = None
        self.partition = None

        self._read_perm()
        self._compute_threshold_speed()

    # ------------------------------------------------------------------------------#

    def _compute_size(self):
        dim = self.layers.size
        if dim == 1:
            self.shape = list(self.full_shape[:2])
            self.physdims = list(self.full_physdims[:2])
        else:
            self.shape = list(self.full_shape[:2]) + [dim]
            thickness = self.full_physdims[2] / self.full_shape[2] * dim
            self.physdims = list(self.full_physdims[:2]) + [thickness]

        self.N = np.prod(self.shape)
        self.n = np.prod(self.shape[:2])

    # ------------------------------------------------------------------------------#

    def _create_mdg(self,):
        self.sd = pp.CartGrid(self.shape, self.physdims)
        self.sd.compute_geometry()

        # it's only one grid but the solver is build on a mdg
        self.mdg = pp.meshing.subdomains_to_mdg([self.sd])

    # ------------------------------------------------------------------------------#

    def _read_perm(self):
        shape = (self.n, self.layers.size)
        perm_xx, perm_yy, perm_zz = np.empty(shape), np.empty(shape), np.empty(shape)
        layers_id = np.empty(shape)

        # get background (without lenses) permeability or porosity as an array
        perm = self.parameters.bg_array

        # add highly permeable lenses to intrinsic permeability
        lenses = self.parameters.lenses

        for pos, layer in enumerate(self.layers):
            # get cells within the lenses
            coords = self.sd.cell_centers
            for lens in lenses:
                cells_lens = [lens["bounds"](coords[0, i], coords[1, i]) for i in range(self.n)]
                perm[cells_lens] = lens["val"]

            # translate porosity [-] into permeability [m2] if needed
            if self.parameters.data_kind == "poro":
                perm = self.kozeny_carman(perm)

            # fill variables to visualize
            perm_xx[:, pos] = perm.copy()
            perm_yy[:, pos] = perm.copy()
            perm_zz[:, pos] = perm.copy()
            layers_id[:, pos] = layer

        # reshape
        shape = self.n*self.layers.size
        perm_xx = perm_xx.reshape(shape, order="F")
        perm_yy = perm_yy.reshape(shape, order="F")
        perm_zz = perm_zz.reshape(shape, order="F")
        self.perm = np.stack((perm_xx, perm_yy, perm_zz)).T # final intrinsic permeability

        self.layers_id = layers_id.reshape(shape, order="F")

    # ------------------------------------------------------------------------------#

    def _compute_threshold_speed(self):
        mu = self.parameters.mu
        beta = self.parameters.beta
        Fo_c = self.parameters.Fo_c
        K_max = np.max(self.perm[:, 0]) # maximum intrinsic permeability [m2]

        self.u_bar = mu/(beta*np.sqrt(K_max)) * Fo_c # threshold speed [m/s]

        print("u_bar =", round(self.u_bar,5), "[m/s]")

    # ------------------------------------------------------------------------------#

    def save_perm(self):
        names = ["log10_perm_xx", "log10_perm_yy", "log10_perm_zz", "layer_id",
                 "perm_xx", "perm_yy", "perm_zz"]

        # for visualization export the intrinsic perm
        for _, d in self.mdg.subdomains(return_data=True):
            d[pp.STATE][names[0]] = np.log10(self.perm[:, 0])
            d[pp.STATE][names[1]] = np.log10(self.perm[:, 1])
            d[pp.STATE][names[2]] = np.log10(self.perm[:, 2])

            d[pp.STATE][names[3]] = self.layers_id

            d[pp.STATE][names[4]] = self.perm[:, 0]
            d[pp.STATE][names[5]] = self.perm[:, 1]
            d[pp.STATE][names[6]] = self.perm[:, 2]

        return names

    # ------------------------------------------------------------------------------#

    def perm_as_dict(self):
        return {"kxx": self.perm[:, 0], "kyy": self.perm[:, 1], "kzz": self.perm[:, 2]}

    # ------------------------------------------------------------------------------#

    def save_forch_vars(self, flux=None):
        names = ["Forchheimer number", "P0_darcy_flux_denormalized", \
                 "P0_darcy_flux_denormalized_norm"]

        # for visualization export the Forchheimer number and denormalized fluxes (*u_bar)
        if flux is None: # no flux is given: only give name of variable to visualize
            return names
        else: # flux is given: compute Forchheimer number and store it
            Fo_c = self.parameters.Fo_c
            u_bar = self.u_bar
            for _, d in self.mdg.subdomains(return_data=True):
                d[pp.STATE][names[0]] = Fo_c * np.linalg.norm(flux, axis=0)
                d[pp.STATE][names[1]] = flux * u_bar
                d[pp.STATE][names[2]] = np.linalg.norm(flux * u_bar, axis=0)

    # ------------------------------------------------------------------------------#

    def kozeny_carman(self, phi):
        # kozeny carman reference parameters
        phi_ref = 0.35           # [-]
        K_ref = 1.0152441851e-9  # [m2]

        return K_ref * \
            np.square(1-phi_ref)/np.power(phi_ref,3) * np.power(phi,3)/np.square(1-phi)
