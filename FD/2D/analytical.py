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
# Creation Date : 2018-12-05 - 22:41:15
"""
-----------

One dimensionnal solution of the propagation and reflection of a Gaussian
pulse.

@author: Cyril Desjouy
"""

import numpy as np
import matplotlib.pylab as pl
from scipy import ifft
from scipy.special import hankel1


def GaussianPulseDR(epsilon, Bx, c0, zm, zs, LOG='no'):
    ''' Analytical Propgation of a gaussion pulse near a rigid boundary.
    epsilon : amplitude of the gaussian
    Bx : width of the gaussian
    c0 : celerity of wave
    zm : location of the microphone
    zs : location of the source
    LOG : if LOG=='log', display complementary informations
    '''

    ext = 10
    Nfreq = 2**12
    Npadd = 2**14
    B = np.sqrt(Bx**2/np.log(2))
    k0 = np.sqrt(2)/B
    fmin = 0.00001
    fmax = k0*c0/(2*np.pi)
    r1 = abs(zm-zs)
    r2 = abs(zm+zs)

    # Axes
    k = np.linspace(2*np.pi*fmin/c0, ext*k0, Nfreq)
    f = np.linspace(fmin, ext*fmax, Nfreq)
    df = f[2] - f[1]
    wtime = np.linspace(0, 1/df, Npadd)

    # Power of the source % omega
    Sw = 1j*k*np.pi*epsilon*B**2*np.exp(-k**2*B**2/4.)/c0

    # Pressure in the freq domain.
    Pdw = -1j*Sw*hankel1(0, k*r1)/4.
    Prw = -1j*Sw*hankel1(0, k*r2)/4.

    Ai = fmax*Nfreq/(2*np.pi)
    Pdt = Ai*ifft(Pdw, Npadd)[::-1]
    Prt = Ai*ifft(Prw, Npadd)[::-1]

    if LOG == 'log':
        print('Max. frequency : ' + repr(fmax))
        pl.figure('Source Strenght')
        pl.subplot(311)
        pl.plot(f, abs(Sw), 'k')
        pl.ylabel('Strength $S_w$')
        pl.subplot(312)
        pl.plot(f, abs(Pdw), 'k', label='Direct')
        pl.plot(f, abs(Prw), color='0.5', label='Reflected')
        pl.legend()
        pl.ylabel(r'Pressure $\tilde{p}(r,w)$')
        pl.subplot(313)
        pl.plot(wtime, Pdt.real/Pdt.real.max(), color='0.8', label='Direct')
        pl.plot(wtime, Prt.real/Prt.real.max(), color='0.5', label='Reflected')
        pl.plot(wtime, Pdt.real/Pdt.real.max() + Prt.real/Prt.real.max(), 'k--', linewidth=4, label='Sum')
        pl.legend()
        pl.ylabel(r'Pressure $\tilde{p}(r,t)$')

    return Pdt+Prt, Pdt, Prt, f, wtime
