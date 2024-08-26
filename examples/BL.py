# PyCompOSE: manages CompOSE tables
# Copyright (C) 2022, David Radice <david.radice@psu.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Tested using: https://compose.obspm.fr/eos/121
#
# This example shows how to import/export CompOSE tables and compute the sound speed
#
# Instructions
# - Create a folder "BL" at the same location as this script
# - Download https://compose.obspm.fr/eos/121 and place it in a folder "BL"
# - Run the script

# %%
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))
from compose.eos import Metadata, Table

# %%
md = Metadata(
    pairs = {
        0: ("e", "electron"),
        1: ("mu", "muon"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
    }
)
eos = Table(md)
eos.read(os.path.join(SCRIPTDIR, "BL"))

# %%
eos.compute_cs2(floor=1e-6)
eos.validate()
eos.shrink_to_valid_nb()
eos.write_hdf5(os.path.join(SCRIPTDIR, "BL", "BL.h5"))
eos.write_lorene(os.path.join(SCRIPTDIR, "BL", "BL.lorene"))
eos.write_txt(os.path.join(SCRIPTDIR, "BL", "BL.txt"))

# %%
print("{} <= nb <= {}".format(eos.nb.min(), eos.nb.max()))
print("{} <= cs2 <= {}".format(eos.thermo["cs2"].min(), eos.thermo["cs2"].max()))

# %%
plt.figure()
plt.loglog(eos.nb, eos.thermo["Q1"][:,0,0]*eos.nb)
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$p\ [{\rm MeV}\ {\rm fm}^{-3}]$")

# %%
plt.figure()
plt.plot(eos.nb, eos.Y["e"][:,0,0], label=r"$Y_e$")
plt.plot(eos.nb, eos.Y["mu"][:,0,0], label=r"$Y_\mu$")
plt.plot(eos.nb, eos.Y["n"][:,0,0], label=r"$Y_n$")
plt.plot(eos.nb, eos.Y["p"][:,0,0], label=r"$Y_p$")
plt.legend()
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")

# %%
plt.figure()
plt.loglog(eos.nb, eos.thermo["cs2"][:,0,0])
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$c_s^2/c^2$")


# %%
plt.show()
