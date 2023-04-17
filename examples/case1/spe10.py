import numpy as np
from scipy.stats import hmean
import porepy as pp

# ------------------------------------------------------------------------------#

class Spe10(object):

# ------------------------------------------------------------------------------#

    def __init__(self, layers, pos_x=None, pos_y=None):
        self.full_shape = np.array([60, 220, 85])
        self.full_physdims = np.array([365.76, 670.56, 51.816])

        if pos_x is not None:
            self.pos_x = np.asarray(pos_x)
        else:
            self.pos_x = np.arange(self.full_shape[0])

        if pos_y is not None:
            self.pos_y = np.asarray(pos_y)
        else:
            self.pos_y = np.arange(self.full_shape[1])

        self.layers = np.sort(np.atleast_1d(layers))

        self.N = 0
        self.n = 0
        self._compute_size()

        self.mdg = None
        self._create_mdg()

        self.perm = None
        self.layers_id = None
        self.partition = None

# ------------------------------------------------------------------------------#

    def _compute_size(self):
        dim = self.layers.size

        x_range = np.linspace(0, self.full_physdims[0], self.full_shape[0]+1)[np.r_[self.pos_x, self.pos_x[-1]+1]]
        y_range = np.linspace(0, self.full_physdims[1], self.full_shape[1]+1)[np.r_[self.pos_y, self.pos_y[-1]+1]]

        bounding_box = {"xmin": x_range[0], "xmax": x_range[-1],
                        "ymin": y_range[0], "ymax": y_range[-1],
                        "zmin": 0, "zmax": 0}
        if dim == 1:
            self.shape = np.hstack((self.pos_x.size, self.pos_y.size))
            self.physdims = np.hstack((x_range[-1], y_range[-1]))
        else:
            self.shape = np.hstack((self.pos_x.size, self.pos_y.size, dim))
            thickness = self.full_physdims[2] / self.full_shape[2] * dim
            self.physdims = np.hstack((x_range[-1], y_range[-1], thickness))
            bounding_box["zmax"] = self.physdims[-1]

        self.N = np.prod(self.shape)
        self.n = np.prod(self.shape[:2])
        self.domain = pp.Domain(bounding_box=bounding_box)

# ------------------------------------------------------------------------------#

    def _create_mdg(self,):
        sd = pp.CartGrid(self.shape, self.domain.bounding_box)
        sd.compute_geometry()

        # it's only one grid but the solver is build on a mdg
        self.mdg = pp.meshing.subdomains_to_mdg([sd])

# ------------------------------------------------------------------------------#

    def read_perm(self, perm_folder):

        shape = (self.n, self.layers.size)
        perm_xx, perm_yy, perm_zz = np.empty(shape), np.empty(shape), np.empty(shape)
        layers_id = np.empty(shape)

        for pos, layer in enumerate(self.layers):
            perm_file = perm_folder + str(layer) + ".tar.gz"
            #perm_file = perm_folder + "small_0.csv"
            perm_layer = np.loadtxt(perm_file, delimiter=",")

            perm_x = perm_layer[:, 0].reshape(self.full_shape[:2], order="F")
            perm_y = perm_layer[:, 1].reshape(self.full_shape[:2], order="F")
            perm_z = perm_layer[:, 2].reshape(self.full_shape[:2], order="F")

            perm_xx[:, pos] = perm_x[self.pos_x, :][:, self.pos_y].flatten(order="F")
            perm_yy[:, pos] = perm_y[self.pos_x, :][:, self.pos_y].flatten(order="F")
            perm_zz[:, pos] = perm_z[self.pos_x, :][:, self.pos_y].flatten(order="F")
            layers_id[:, pos] = layer

        shape = self.n*self.layers.size
        perm_xx = perm_xx.reshape(shape, order="F")
        perm_yy = perm_yy.reshape(shape, order="F")
        perm_zz = perm_zz.reshape(shape, order="F")
        self.perm = np.stack((perm_xx, perm_yy, perm_zz)).T

        self.layers_id = layers_id.reshape(shape, order="F")

# ------------------------------------------------------------------------------#

    def save_perm(self):

        names = ["log10_perm_xx", "log10_perm_yy", "log10_perm_zz", "layer_id",
                 "perm_xx", "perm_yy", "perm_zz"]

        # for visualization export the perm and layer id
        for _, d in self.mdg.subdomains(return_data=True):

            d[pp.STATE][names[0]] = np.log10(self.perm[:, 0])
            d[pp.STATE][names[1]] = np.log10(self.perm[:, 1])
            d[pp.STATE][names[2]] = np.log10(self.perm[:, 2])

            d[pp.STATE][names[3]] = self.layers_id

            d[pp.STATE][names[4]] = self.perm[:, 0] * pp.DARCY
            d[pp.STATE][names[5]] = self.perm[:, 1] * pp.DARCY
            d[pp.STATE][names[6]] = self.perm[:, 2] * pp.DARCY

        return names

# ------------------------------------------------------------------------------#

    def perm_as_dict(self):
        return {"kxx": self.perm[:, 0], "kyy": self.perm[:, 1], "kzz": self.perm[:, 2]}

# ------------------------------------------------------------------------------#
