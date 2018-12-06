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
# Creation Date : 2018-12-05 - 22:31:11
"""
-----------
DOCSTRING

@author: Cyril Desjouy
"""

import matplotlib.pyplot as plt
from analytical import GaussianPulseDR
from fdtd import FDTD
from param import Param

if __name__ == "__main__":

    # Numerical solution (FDTD)
    Param.update_param()
    fdtd = FDTD(Param)
    rhou, rhov, pac = fdtd.run()

    # Analytical solution
    Pt, _, _, _, wtime = GaussianPulseDR(Param.epsilon, Param.BWx, Param.c0,
                                         Param.z[int(Param.zP)], Param.zS)

    # Figures
    plt.figure('Acoustic variables', figsize=(14, 6))

    plt.subplot(131)
    plt.imshow(pac - Param.p0, interpolation='nearest')
    plt.plot(Param.zP, Param.xP, 'ro')
    plt.axis([0, Param.nbz, 0, Param.nbx])
    plt.title(r'$p_a$', fontsize=22)
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(rhou/Param.rho0, interpolation='nearest')
    plt.plot(Param.zP, Param.xP, 'ro')
    plt.title(r'$v_x$', fontsize=22)
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(rhov/Param.rho0, interpolation='nearest')
    plt.plot(Param.zP, Param.xP, 'ro')
    plt.title(r'$v_z$', fontsize=22)
    plt.colorbar()

    plt.figure('Microphones', figsize=(13, 3))
    plt.plot(fdtd.t, fdtd.probe/max(fdtd.probe), 'k', linewidth=3, label='FDTD')
    plt.plot(wtime, Pt.real/Pt.real.max(), 'r--', linewidth=3, label='Theoretical')
    plt.xlim([Param.t.min(), Param.t.max()])
    plt.legend()
    plt.grid()
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$p_{ac}$ [Pa]')

    plt.show()
