import sys

def write_network_pvd(file_name, folder_name, iterations):
    o_file = open(folder_name + "net_" + file_name + ".pvd", "w")
    b = "LittleEndian" if sys.byteorder == "little" else "BigEndian"
    c = ' compressor="vtkZLibDataCompressor"'
    header = (
        '<?xml version="1.0"?>\n'
        + '<VTKFile type="Collection" version="0.1" '
        + 'byte_order="%s"%s>\n' % (b, c)
        + "<Collection>\n"
    )
    o_file.write(header)
    fm = '\t<DataSet group="" part="" timestep="%f" file="%s"/>\n'

    for it in iterations:
        o_file.write(fm % (it, make_file_name(file_name, it) + ".vtu"))

    o_file.write("</Collection>\n" + "</VTKFile>")
    o_file.close()

def make_file_name(file_name, iteration):
    padding = 6
    return "net_" + file_name + "_" + str(iteration).zfill(padding)


