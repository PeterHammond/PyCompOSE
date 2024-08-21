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

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

def find_valid_region(arr):
    """
    Utility function.

    Finds the largest contiguous range (i0, i1) in which
        np.all(arr[i0:i1]) == True
    """
    def regions(arr):
        i0 = 0
        while i0 < len(arr):
            if arr[i0] == True:
                for i1 in range(i0, len(arr)):
                    if arr[i1] == False:
                        yield (i0, i1)
                        i0 = i1 + 1
                        break
                    elif i1 == len(arr)-1:
                        yield (i0, len(arr))
                        i0 = len(arr)
                        break
            else:
                i0 += 1

    rlist = list(regions(arr))
    irmax, lrmax = 0, 0
    for ir, r in enumerate(rlist):
        myl = r[1] - r[0]
        if myl > lrmax:
            lrmax = myl
            irmax = ir

    return rlist[irmax]

def interpolator(x, y, **kwargs):
    """
    Custom 1d interpolator
    """
    return CubicSpline(x, y, **kwargs)

def find_temp_given_ent(t, yq, S, S0, options={'xatol': 1e-2, 'maxiter': 100}):
    """
    Find the temperature such that S(T, Yq) = S0 for each ye

    * t  : 1d grid of temperatures
    * yq : 1d grid of Ye
    * S  : 2d array of entropy S[iye,itemp]
    * S0 : wanted entropy

    options are passed to `scipy.optimize.minimize_scalar`
    """
    tout = np.zeros_like(ye1d)
    for iyq in range(yq.shape[0]):
        f = interpolator(t, (S[iyq,:] - S0)**2)
        res = minimize_scalar(f, bounds=(t[0], t[-1]), method='bounded',
            options=options)
        tout[iyq] = res.x
    return tout

def find_beta_eq(yq, mu_l, options={'xatol': 1e-6, 'maxiter': 100}):
    """
    Find the neutrino-less beta equilibrium ye for each point
    in a 1D table tab(ye).

    Beta equilibrium is found from the condition

    mu_l = 0

    * yq   : charge fraction
    * mu_l : lepton chemical potential as a function of yq

    options are passed to `scipy.optimize.minimize_scalar`
    """
    # These cases have beta-equilibrium out of the table
    if np.all(mu_l > 0):
        return yq[0]
    if np.all(mu_l < 0):
        return yq[-1]

    f = interpolator(yq, mu_l**2)
    res = minimize_scalar(f, bounds=(yq[0], yq[-1]), method='bounded',
    options=options)
    return res.x

def read_micro_composite_index(Ki):
    """
    Convert a composite index Ki into a tuple (name, desc) for eos.micro quantites.
    """


    # Abridged copy of table 3.3 in CompOSE manual v3
    dense_matter_fermions = {0:  ("e",  "electron"),
                             1:  ("mu", "muon"),
                             10: ("n",  "neutron"),
                             11: ("p",  "proton")}

    # Abridged copy of table 7.5 in CompOSE manual v3
    microscopic_quantites = {40: ("mL_{0:s}", "Effective {1:s} Landau mass with respect to particle mass: mL_{0:s} / m_{0:s} []"),
                             41: ("mD_{0:s}", "Effective {1:s} Dirac mass with respect to particle mass: mD_{0:s} / m_{0:s} []"),
                             50: ("U_{0:s}", "Non-relativistic {1:s} single-particle potential: U_{0:s} [MeV]"),
                             51: ("V_{0:s}", "Relativistic {1:s} vector self-energy: V_{0:s} [MeV]"),
                             52: ("S_{0:s}", "Relativistic {1:s} scalar self-energy: S_{0:s} [MeV]")}

    Ii = Ki//1000
    Ji = Ki - 1000*Ii

    particle_names = dense_matter_fermions[Ii]
    variable_symbol, variable_description = microscopic_quantites[Ji]
    variable_names = (variable_symbol.format(*particle_names), variable_description.format(*particle_names))

    return variable_names

def NQT_log(x):
    assert(np.all(x>0.0))
    # Positive definite x gives mantissa in range 0.5-1.0, we want 1.0-2.0 (then subtract 1)
    mantissa, exponent = np.frexp(x)
    mantissa = 2*mantissa - 1
    exponent -= 1 # We doubled the mantissa, so must subtract 1 from the exponent.
    return mantissa + exponent

def NQT_exp(x):
    # Gives us fractional and integer part, mantissa 0.0-1.0
    exponent, mantissa = np.divmod(x,1)
    mantissa = 0.5*(mantissa+1)
    exponent = np.array(exponent,dtype=int) + 1 # opposite of above
    return np.ldexp(mantissa,exponent)