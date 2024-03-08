import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=25)

params = {
    "text.latex.preamble": r"\usepackage{bm,amsmath,siunitx}\newcommand{\sib}[1]{[\si{#1}]}"
}
plt.rcParams.update(params)
mpl.rcParams["axes.linewidth"] = 1.5


def read_org(folder, file_name):
    file_name = folder + file_name
    data = np.loadtxt(file_name, delimiter="|", dtype=str)

    data = data[:, 1:-1]

    h_p = data[1:-1, 6].astype(float)
    h_v = data[1:-1, 12].astype(float)
    fraction_cells = data[2:, 23].astype(float)
    inv_e = data[1:-1, 0].astype(float)

    return inv_e, h_p, h_v, fraction_cells


def read_txt(folder, file_name):
    file_name = folder + file_name
    data = np.loadtxt(file_name, delimiter=",", dtype=float)

    inv_e = data[0, :]
    h_p = data[1, :]
    h_v = data[2, :]
    fraction_cells = data[3, :]

    return inv_e, h_p, h_v, fraction_cells


def main(folder, file_name, file_out, legend):

    size = 7
    fig, ax1 = plt.subplots(figsize=(size * 1.5, size))
    ax2 = ax1.twinx()

    line_type = ["-", "--", "*--"]
    lns = []

    for fname, ltype, leg in zip(file_name, line_type, legend):
        file_type = fname[-3:]

        if file_type == "org":
            inv_e, h_p, h_v, fraction_cells = read_org(folder, fname)
        elif file_type == "txt":
            inv_e, h_p, h_v, fraction_cells = read_txt(folder, fname)

        l1 = ax1.loglog(inv_e, h_p, ltype + "r", label=leg + " - $err_p$")
        l2 = ax1.loglog(inv_e, h_v, ltype + "b", label=leg + " - $err_q$")
        l3 = ax2.semilogx(
            inv_e, fraction_cells, ltype + "g", label=leg + " - $F$-cells"
        )

        lns += l1 + l2 + l3

    # labs = [l.get_label() for l in lns]
    # ax1.legend(lns, labs, loc="best")

    ax1.set_xlabel("$1/E$")
    ax1.set_ylabel("$L^2-err \sib{\cdot}$")
    ax2.set_ylabel("$\sharp D-F$ cells \sib{\percent}")

    plt.show()

    fig.savefig(folder + file_out + ".pdf", bbox_inches="tight")

    # plot the legend
    handles1, labels1 = [
        (a + b)
        for a, b in zip(
            ax1.get_legend_handles_labels(), ax1.get_legend_handles_labels()
        )
    ]
    handles2, labels2 = [
        (a + b)
        for a, b in zip(
            ax2.get_legend_handles_labels(), ax2.get_legend_handles_labels()
        )
    ]
    handles = handles1 + handles2
    labels = labels1 + labels2
    labels, mask = np.unique(labels, return_index=True)
    reorder = [0, 1, 2, 6, 7, 8, 3, 4, 5]
    labels = [labels[i] for i in reorder]
    handles = [handles[i] for i in mask[reorder]]
    # legend
    # handles = handles[:12]
    fig, ax = plt.subplots(figsize=(25, 10))
    for h, l in zip(handles, labels):
        ax.plot(np.zeros(1), label=l)

    ax.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(-0.1, -0.65))

    filename = folder + file_out + "_legend.pdf"
    fig.savefig(filename, bbox_inches="tight")
    plt.gcf().clear()

    os.system("pdfcrop --margins '0 -750 0 0' " + filename + " " + filename)
    os.system("pdfcrop " + filename + " " + filename)


if __name__ == "__main__":

    folder = "./"  # "./examples/case1/"
    file_name = ["output35_10.txt", "output35_50.txt", "output35_200.txt"]
    legend = ["$q_{in} = 10$", "$q_{in} = 50$", "$q_{in} = 200$"]
    main(folder, file_name, "err_case1", legend)
