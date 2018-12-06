#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2016-2018 Cyril Desjouy <cyril.desjouy@univ-lemans.fr>
#
# This file is part of pyLEEx
#
# pyLEEx is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyLEEx is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyLEEx. If not, see <http://www.gnu.org/licenses/>.
#
#
# Creation Date : 2018-12-04 - 22:53:29
"""
-----------

Parameters of the FDTD simulation.

@author: Cyril Desjouy
"""

import numpy as np
from coefficients import a3std, a5std, a7std, a7drp


class Param:
    """
    Geometrical and thermophysical parameters
    """

    # Parameters
    gamma = 1.4
    rho0 = 1.22
    c0 = 340.
    p0 = rho0*c0**2/gamma
    dx = 0.0001                     # Spatial step
    dz = dx                         # Spatial step
    one_dx, one_dz = 1/dx, 1/dz     # Optimisation
    CFL = 0.5                       # CFL
    dt = dx*CFL/c0                  # Timestep
    epsilon = 1e5                   # amplitude
    nbx, nbz = 256, 128             # Grid
    ix0, iz0 = int(nbx/2), 0               # Grid origin
    xS, zS = 0.0, dx*nbz/2.         # Source location
    xP, zP = nbx/2, 2*nbz/8         # Microphone location
    BWx = 0.0005                    # Spacial bandwidth
    Nit_out = 10                    # Number of iteration between each output
    tend = 30e-6                    # Compute until 'tend' seconds
    Nit = int(tend/dt)              # Number of iteration

    # Coefficents of the FD schemes
    a3c, a3d = a3std()
    a5c, a5d = a5std()
    a7c, a7d = a7std()
    a7cT, a7dT = a7drp()
    rka = np.array([0.5, 0.5, 1.])
    rkb = np.array([1/6., 1/3., 1/3., 1/6.])

    @classmethod
    def update_param(cls):
        """
        Update values if class attributes are changed from outside.
        """
        Param.Nit = int(Param.tend/Param.dt)
        Param.one_dx, Param.one_dz = 1/Param.dx, 1/Param.dz
        Param.p0 = Param.rho0*Param.c0**2/Param.gamma
        Param.dt = Param.dx*Param.CFL/Param.c0
        Param.t = np.arange(0, Param.Nit*Param.dt, Param.dt)
        Param.x = (np.arange(0, Param.nbx) - Param.ix0)*Param.dx
        Param.z = (np.arange(0, Param.nbz) - Param.iz0)*Param.dz


    @classmethod
    def init_fields(cls):
        """
        Initialize conservative variables
        """
        rhou1 = np.zeros((Param.nbx, Param.nbz))
        rhov1 = np.zeros((Param.nbx, Param.nbz))
        p1 = np.zeros((Param.nbx, Param.nbz))

        for iz in range(0, Param.nbz):
            p1[:, iz] = Param.p0 + \
                    Param.epsilon*np.exp(-np.log(2)*((Param.x-Param.xS)**2 +
                                                     (Param.z[iz]-Param.zS)**2)/Param.BWx**2)

        return rhou1, rhov1, p1
