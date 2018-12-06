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
# Creation Date : 2018-12-05 - 22:02:35
"""
-----------
DOCSTRING

@author: Cyril Desjouy
"""

import os
import time
import numpy as np
from efluxes import EulerianFluxes


class FDTD:
    """
    Finite Difference Time Domain Method
    """

    def __init__(self, param, stencil='7pts_o', filtering='yes'):

        self.param = param
        self.probe = []
        self.rhou, self.rhov, self.pac = param.init_fields()
        self.efluxes = EulerianFluxes(param, stencil, filtering).rk4
        if not os.path.isdir('results/'):
            print("Creating 'results' directory...")
            os.mkdir('results/')

    def __getattr__(self, attr):
        return getattr(self.param, attr)

    def run(self):
        """
        Run FDTD Simulation
        """
        # boucle principale
        perf = []
        print('Start main loop')
        for it in range(self.Nit):
            perf_i = time.perf_counter()
            self.eulerian_fluxes()

            # Residual pressure
            residual = np.sqrt(np.sum((self.pac-self.p0)**2)/(self.nbx*self.nbz))

            # Probe
            self.probe.append(self.pac[int(self.xP), int(self.zP)] - self.p0)

            # Performance
            perf.append(time.perf_counter() - perf_i)

            if abs(residual) > 100*self.epsilon:
                print('Divergence ! Stopped at it = ' + repr(it))
                break
            if np.mod(it, self.Nit_out) == 0:
                print('it: {:4} | res: {:8.2f} | time/it: {:5.3f} s.'.format(it,
                                                                             residual,
                                                                             np.mean(perf)))
                np.savez_compressed('results/it{}'.format(it),
                                    p=self.pac,
                                    vx=self.rhou/self.rho0,
                                    vz=self.rhov/self.rho0,
                                    x=self.x, z=self.z)
                perf = []
        print('End main loop ')

        return self.rhou, self.rhov, self.pac

    def eulerian_fluxes(self):
        """ Eulerian fluxes """

        self.rhou, self.rhov, self.pac = self.efluxes(self.rhou, self.rhov, self.pac)
