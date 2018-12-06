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

import numpy as np


class EulerianFluxes:
    """
    Compute Eulerian fluxes
    """

    def __init__(self, param, stencil, filtering):

        self.param = param

        self.prk = np.array([])
        self.rhourk = np.array([])
        self.rhovrk = np.array([])

        self.Kpu = np.array([])
        self.Kpv = np.array([])
        self.Krhou = np.array([])
        self.Krhov = np.array([])

        self.Ep = np.array([])
        self.Fp = np.array([])

        self.Erhou = np.array([])
        self.Frhov = np.array([])

        if stencil == '3pts':
            self.cin = self.cin3
            self.P = np.array(list(range(-1, self.nbx)) + list(range(2)))
        elif stencil == '5pts':
            self.cin = self.cin5
            self.P = np.array(list(range(-2, self.nbx)) + list(range(3)))
        elif stencil == '7pts':
            self.cin = self.cin7
            self.P = np.array(list(range(-3, self.nbx)) + list(range(4)))
        elif stencil == '7pts_o':
            self.cin = self.cin7
            self.P = np.array(list(range(-3, self.nbx)) + list(range(4)))
            self.a7c = self.a7cT
            self.a7d = self.a7dT

    def __getattr__(self, attr):
        return getattr(self.param, attr)

    def rk4(self, rhou, rhov, pac):
        """
        Time advance of the solution with RK4 algorithm (4th order)
        """

        p2 = np.zeros_like(pac)
        rhou2 = np.zeros_like(rhou)
        rhov2 = np.zeros_like(rhov)

        self.prk = pac.copy()
        self.rhourk = rhou.copy()
        self.rhovrk = rhov.copy()

        for i in range(4):

            self.cin()

            if i < 3:
                self.prk = pac - self.dt*self.rka[i]*(self.Kpu + self.Kpv)
                self.rhourk = rhou - self.dt*self.rka[i]*self.Krhou
                self.rhovrk = rhov - self.dt*self.rka[i]*self.Krhov

                # CL
                self.rhourk[:, 0] = 0.
                self.rhovrk[:, 0] = 0.
                self.rhourk[:, -1] = 0.
                self.rhovrk[:, -1] = 0.

            p2 += self.dt*self.rkb[i]*(self.Kpu + self.Kpv)
            rhou2 += self.dt*self.rkb[i]*self.Krhou
            rhov2 += self.dt*self.rkb[i]*self.Krhov

        p2 = pac - p2
        rhou2 = rhou - rhou2
        rhov2 = rhov - rhov2

        # CL
        rhou2[:, 0] = 0.
        rhov2[:, 0] = 0.
        rhou2[:, -1] = 0.
        rhov2[:, -1] = 0.

        return rhou2, rhov2, p2


    def init_derivatives(self):
        """
        Initialize derivatives
        """

        # Initialization of derivatives
        self.Kpu = np.zeros_like(self.prk)
        self.Kpv = np.zeros_like(self.prk)
        self.Krhou = np.zeros_like(self.prk)
        self.Krhov = np.zeros_like(self.prk)
        # dE/dx term
        self.Ep = self.rhourk*self.c0**2
        self.Erhou = self.prk.copy()
        # dF/dz term
        self.Fp = self.rhovrk*self.c0**2
        self.Frhov = self.prk.copy()


    def cin3(self):
        """
        Compute derivatives with a 3 points FD scheme
        """
        self.init_derivatives()

        # Following x direction : 3 points centered schem
        for ix in range(self.nbx):
            self.Kpu[ix, :] = (self.a3c[1]*(self.Ep[self.P[1+ix+1], :] -
                                            self.Ep[self.P[1+ix-1], :]))*self.one_dx
            self.Krhou[ix, :] = (self.a3c[1]*(self.Erhou[self.P[1+ix+1], :] -
                                              self.Erhou[self.P[1+ix-1], :]))*self.one_dx

        # Following Z direction : 5 points uncentered scheme
        iz = 0
        self.Kpv[:, iz] = (self.a3d[1]*self.Fp[:, 0] +
                           self.a3d[2]*self.Fp[:, 1] +
                           self.a3d[3]*self.Fp[:, 2])*self.one_dz
        self.Krhov[:, iz] = (self.a3d[1]*self.Frhov[:, 0] +
                             self.a3d[2]*self.Frhov[:, 1] +
                             self.a3d[3]*self.Frhov[:, 2])*self.one_dz

        iz = -1
        self.Kpv[:, iz] = -(self.a3d[1]*self.Fp[:, -1] +
                            self.a3d[2]*self.Fp[:, -2] +
                            self.a3d[3]*self.Fp[:, -3])*self.one_dz
        self.Krhov[:, iz] = -(self.a3d[1]*self.Frhov[:, -1] +
                              self.a3d[2]*self.Frhov[:, -2] +
                              self.a3d[3]*self.Frhov[:, -3])*self.one_dz

        # Following z direction : 5 points centered scheme
        for iz in range(1, self.nbz-1):
            self.Kpv[:, iz] = self.a3c[1]*(self.Fp[:, iz+1] -
                                           self.Fp[:, iz-1])*self.one_dz
            self.Krhov[:, iz] = self.a3c[1]*(self.Frhov[:, iz+1] -
                                             self.Frhov[:, iz-1])*self.one_dz


    def cin5(self):
        """
        Compute derivatives with a 5 points FD scheme
        """

        self.init_derivatives()

        # Following x direction : 5 points centered schem
        for ix in range(self.nbx):
            self.Kpu[ix, :] = (self.a5c[1]*(self.Ep[self.P[2+ix+1], :] -
                                            self.Ep[self.P[2+ix-1], :]) +
                               self.a5c[2]*(self.Ep[self.P[2+ix+2], :] -
                                            self.Ep[self.P[2+ix-2], :]))*self.one_dx
            self.Krhou[ix, :] = (self.a5c[1]*(self.Erhou[self.P[2+ix+1], :] -
                                              self.Erhou[self.P[2+ix-1], :]) +
                                 self.a5c[2]*(self.Erhou[self.P[2+ix+2], :] -
                                              self.Erhou[self.P[2+ix-2], :]))*self.one_dx

        # Following Z direction : 5 points uncentered scheme
        iz = 0
        self.Kpv[:, iz] = (self.a5d[1, 1]*self.Fp[:, 0] +
                           self.a5d[1, 2]*self.Fp[:, 1] +
                           self.a5d[1, 3]*self.Fp[:, 2] +
                           self.a5d[1, 4]*self.Fp[:, 3] +
                           self.a5d[1, 5]*self.Fp[:, 4])*self.one_dz
        self.Krhov[:, iz] = (self.a5d[1, 1]*self.Frhov[:, 0] +
                             self.a5d[1, 2]*self.Frhov[:, 1] +
                             self.a5d[1, 3]*self.Frhov[:, 2] +
                             self.a5d[1, 4]*self.Frhov[:, 3] +
                             self.a5d[1, 5]*self.Frhov[:, 4])*self.one_dz

        iz = 1
        self.Kpv[:, iz] = (self.a5d[2, 1]*self.Fp[:, 0] +
                           self.a5d[2, 2]*self.Fp[:, 1] +
                           self.a5d[2, 3]*self.Fp[:, 2] +
                           self.a5d[2, 4]*self.Fp[:, 3] +
                           self.a5d[2, 5]*self.Fp[:, 4])*self.one_dz
        self.Krhov[:, iz] = (self.a5d[2, 1]*self.Frhov[:, 0] +
                             self.a5d[2, 2]*self.Frhov[:, 1] +
                             self.a5d[2, 3]*self.Frhov[:, 2] +
                             self.a5d[2, 4]*self.Frhov[:, 3] +
                             self.a5d[2, 5]*self.Frhov[:, 4])*self.one_dz

        iz = -1
        self.Kpv[:, iz] = - (self.a5d[1, 1]*self.Fp[:, -1] +
                             self.a5d[1, 2]*self.Fp[:, -2] +
                             self.a5d[1, 3]*self.Fp[:, -3] +
                             self.a5d[1, 4]*self.Fp[:, -4] +
                             self.a5d[1, 5]*self.Fp[:, -5])*self.one_dz
        self.Krhov[:, iz] = - (self.a5d[1, 1]*self.Frhov[:, -1] +
                               self.a5d[1, 2]*self.Frhov[:, -2] +
                               self.a5d[1, 3]*self.Frhov[:, -3] +
                               self.a5d[1, 4]*self.Frhov[:, -4] +
                               self.a5d[1, 5]*self.Frhov[:, -5])*self.one_dz

        iz = -2
        self.Kpv[:, iz] = - (self.a5d[2, 1]*self.Fp[:, -1] +
                             self.a5d[2, 2]*self.Fp[:, -2] +
                             self.a5d[2, 3]*self.Fp[:, -3] +
                             self.a5d[2, 4]*self.Fp[:, -4] +
                             self.a5d[2, 5]*self.Fp[:, -5])*self.one_dz
        self.Krhov[:, iz] = - (self.a5d[2, 1]*self.Frhov[:, -1] +
                               self.a5d[2, 2]*self.Frhov[:, -2] +
                               self.a5d[2, 3]*self.Frhov[:, -3] +
                               self.a5d[2, 4]*self.Frhov[:, -4] +
                               self.a5d[2, 5]*self.Frhov[:, -5])*self.one_dz

        # Following z direction : 5 points centered scheme
        for iz in range(2, self.nbz-2):
            self.Kpv[:, iz] = (self.a5c[1]*(self.Fp[:, iz+1] - self.Fp[:, iz-1]) +
                               self.a5c[2]*(self.Fp[:, iz+2] - self.Fp[:, iz-2]))*self.one_dz
            self.Krhov[:, iz] = (self.a5c[1]*(self.Frhov[:, iz+1] - self.Frhov[:, iz-1]) +
                                 self.a5c[2]*(self.Frhov[:, iz+2] - self.Frhov[:, iz-2]))*self.one_dz


    def cin7(self):
        """
        Compute derivatives with a 7 points FD scheme
        """

        self.init_derivatives()

        # Following x direction : 5 points centered schem
        for ix in range(self.nbx):
            self.Kpu[ix, :] = (self.a7c[1]*(self.Ep[self.P[3+ix+1], :] -
                                            self.Ep[self.P[3+ix-1], :]) +
                               self.a7c[2]*(self.Ep[self.P[3+ix+2], :] -
                                            self.Ep[self.P[3+ix-2], :]) +
                               self.a7c[3]*(self.Ep[self.P[3+ix+3], :] -
                                            self.Ep[self.P[3+ix-3], :]))*self.one_dx
            self.Krhou[ix, :] = (self.a7c[1]*(self.Erhou[self.P[3+ix+1], :] -
                                              self.Erhou[self.P[3+ix-1], :]) +
                                 self.a7c[2]*(self.Erhou[self.P[3+ix+2], :] -
                                              self.Erhou[self.P[3+ix-2], :]) +
                                 self.a7c[3]*(self.Erhou[self.P[3+ix+3], :] -
                                              self.Erhou[self.P[3+ix-3], :]))*self.one_dx

        # Bords : schéma décentré sur 11 points de iz = 0 à 4 et de nbz-4 a nbz:
        iz = 0
        self.Kpv[:, iz] = (self.a7d[1, 1]*self.Fp[:, 0] + self.a7d[1, 2]*self.Fp[:, 1] +
                           self.a7d[1, 3]*self.Fp[:, 2] + self.a7d[1, 4]*self.Fp[:, 3] +
                           self.a7d[1, 5]*self.Fp[:, 4] + self.a7d[1, 6]*self.Fp[:, 5] +
                           self.a7d[1, 7]*self.Fp[:, 6])*self.one_dz
        self.Krhov[:, iz] = (self.a7d[1, 1]*self.Frhov[:, 0] + self.a7d[1, 2]*self.Frhov[:, 1] +
                             self.a7d[1, 3]*self.Frhov[:, 2] + self.a7d[1, 4]*self.Frhov[:, 3] +
                             self.a7d[1, 5]*self.Frhov[:, 4] + self.a7d[1, 6]*self.Frhov[:, 5] +
                             self.a7d[1, 7]*self.Frhov[:, 6])*self.one_dz

        iz = 1
        self.Kpv[:, iz] = (self.a7d[2, 1]*self.Fp[:, 0] + self.a7d[2, 2]*self.Fp[:, 1] +
                           self.a7d[2, 3]*self.Fp[:, 2] + self.a7d[2, 4]*self.Fp[:, 3] +
                           self.a7d[2, 5]*self.Fp[:, 4] + self.a7d[2, 6]*self.Fp[:, 5] +
                           self.a7d[2, 7]*self.Fp[:, 6])*self.one_dz
        self.Krhov[:, iz] = (self.a7d[2, 1]*self.Frhov[:, 0] + self.a7d[2, 2]*self.Frhov[:, 1] +
                             self.a7d[2, 3]*self.Frhov[:, 2] + self.a7d[2, 4]*self.Frhov[:, 3] +
                             self.a7d[2, 5]*self.Frhov[:, 4] + self.a7d[2, 6]*self.Frhov[:, 5] +
                             self.a7d[2, 7]*self.Frhov[:, 6])*self.one_dz
        iz = 2
        self.Kpv[:, iz] = (self.a7d[3, 1]*self.Fp[:, 0] + self.a7d[3, 2]*self.Fp[:, 1] +
                           self.a7d[3, 3]*self.Fp[:, 2] + self.a7d[3, 4]*self.Fp[:, 3] +
                           self.a7d[3, 5]*self.Fp[:, 4] + self.a7d[3, 6]*self.Fp[:, 5] +
                           self.a7d[3, 7]*self.Fp[:, 6])*self.one_dz
        self.Krhov[:, iz] = (self.a7d[3, 1]*self.Frhov[:, 0] + self.a7d[3, 2]*self.Frhov[:, 1] +
                             self.a7d[3, 3]*self.Frhov[:, 2] + self.a7d[3, 4]*self.Frhov[:, 3] +
                             self.a7d[3, 5]*self.Frhov[:, 4] + self.a7d[3, 6]*self.Frhov[:, 5] +
                             self.a7d[3, 7]*self.Frhov[:, 6])*self.one_dz

        iz = -1
        self.Kpv[:, iz] = - (self.a7d[1, 1]*self.Fp[:, -1] + self.a7d[1, 2]*self.Fp[:, -2] +
                             self.a7d[1, 3]*self.Fp[:, -3] + self.a7d[1, 4]*self.Fp[:, -4] +
                             self.a7d[1, 5]*self.Fp[:, -5] + self.a7d[1, 6]*self.Fp[:, -6] +
                             self.a7d[1, 7]*self.Fp[:, -7])*self.one_dz
        self.Krhov[:, iz] = - (self.a7d[1, 1]*self.Frhov[:, -1] + self.a7d[1, 2]*self.Frhov[:, -2] +
                               self.a7d[1, 3]*self.Frhov[:, -3] + self.a7d[1, 4]*self.Frhov[:, -4] +
                               self.a7d[1, 5]*self.Frhov[:, -5] + self.a7d[1, 6]*self.Frhov[:, -6] +
                               self.a7d[1, 7]*self.Frhov[:, -7])*self.one_dz

        iz = -2
        self.Kpv[:, iz] = - (self.a7d[2, 1]*self.Fp[:, -1] + self.a7d[2, 2]*self.Fp[:, -2] +
                             self.a7d[2, 3]*self.Fp[:, -3] + self.a7d[2, 4]*self.Fp[:, -4] +
                             self.a7d[2, 5]*self.Fp[:, -5] + self.a7d[2, 6]*self.Fp[:, -6] +
                             self.a7d[2, 7]*self.Fp[:, -7])*self.one_dz
        self.Krhov[:, iz] = - (self.a7d[2, 1]*self.Frhov[:, -1] + self.a7d[2, 2]*self.Frhov[:, -2] +
                               self.a7d[2, 3]*self.Frhov[:, -3] + self.a7d[2, 4]*self.Frhov[:, -4] +
                               self.a7d[2, 5]*self.Frhov[:, -5] + self.a7d[2, 6]*self.Frhov[:, -6] +
                               self.a7d[2, 7]*self.Frhov[:, -7])*self.one_dz

        iz = -3
        self.Kpv[:, iz] = - (self.a7d[3, 1]*self.Fp[:, -1] + self.a7d[3, 2]*self.Fp[:, -2] +
                             self.a7d[3, 3]*self.Fp[:, -3] + self.a7d[3, 4]*self.Fp[:, -4] +
                             self.a7d[3, 5]*self.Fp[:, -5] + self.a7d[3, 6]*self.Fp[:, -6] +
                             self.a7d[3, 7]*self.Fp[:, -7])*self.one_dz
        self.Krhov[:, iz] = - (self.a7d[3, 1]*self.Frhov[:, -1] + self.a7d[3, 2]*self.Frhov[:, -2] +
                               self.a7d[3, 3]*self.Frhov[:, -3] + self.a7d[3, 4]*self.Frhov[:, -4] +
                               self.a7d[3, 5]*self.Frhov[:, -5] + self.a7d[3, 6]*self.Frhov[:, -6] +
                               self.a7d[3, 7]*self.Frhov[:, -7])*self.one_dz


        # Calcul intérieur suivant z : schéma centré sur 11 points de iz = 6 à nbz-5
        for iz in range(3, self.nbz-3):
            self.Kpv[:, iz] = (self.a7c[1]*(self.Fp[:, iz+1] - self.Fp[:, iz-1]) +
                               self.a7c[2]*(self.Fp[:, iz+2] - self.Fp[:, iz-2]) +
                               self.a7c[3]*(self.Fp[:, iz+3] - self.Fp[:, iz-3]))*self.one_dz
            self.Krhov[:, iz] = (self.a7c[1]*(self.Frhov[:, iz+1] - self.Frhov[:, iz-1]) +
                                 self.a7c[2]*(self.Frhov[:, iz+2] - self.Frhov[:, iz-2]) +
                                 self.a7c[3]*(self.Frhov[:, iz+3] - self.Frhov[:, iz-3]))*self.one_dz
