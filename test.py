import cupy as cp
import numpy as np
import math
import cupy.cuda.nvtx as nvtx
from MDAnalysis.lib.formats.libdcd import DCDFile
from timeit import default_timer as timer


def dcdreadhead(infile):
    nconf   = infile.n_frames
    _infile = infile.header
    numatm  = _infile['natoms']
    return numatm, nconf

def dcdreadframe(infile, numatm, nconf):

    d_x = np.zeros(numatm * nconf, dtype=np.float64)
    d_y = np.zeros(numatm * nconf, dtype=np.float64)
    d_z = np.zeros(numatm * nconf, dtype=np.float64)

    for i in range(nconf):
        data = infile.readframes(i, i+1)
        box = data[1]
        atomset = data[0][0]
        xbox = round(box[0][0], 8)
        ybox = round(box[0][2],8)
        zbox = round(box[0][5], 8)

        for row in range(numatm):
            d_x[i * numatm + row] = round(atomset[row][0], 8) # 0 is column
            d_y[i * numatm + row] = round(atomset[row][1], 8)  # 1 is column
            d_z[i * numatm + row] = round(atomset[row][2], 8)  # 2 is column

    return xbox, ybox, zbox, d_x, d_y, d_z
