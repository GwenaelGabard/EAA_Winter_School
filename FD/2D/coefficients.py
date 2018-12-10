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

Coefficients of the finite difference schemes :

    * a3std -- 3 points Standard scheme
    * a5std -- 5 points Standard scheme
    * a7std -- 7 points Standard scheme
    * a7drp -- 7 points Dispersion-Relation-Preserving scheme (By CKW TAM - 1993)


@author: Cyril Desjouy
"""


import numpy as np


def a3std():
    ''' 3 points standard coefficients '''

    a3c = np.zeros(3)
    a3c[0] = 0.
    a3c[1] = 0.5
    a3c[-1] = -a3c[1]

    a3d = np.zeros(4)
    a3d[0] = 0
    a3d[1] = -1.5
    a3d[2] = 2
    a3d[3] = -0.5

    return a3c, a3d


def a5std():
    ''' 5 points standard coefficients '''

    a5c = np.zeros(3)
    a5c[0] = 0.
    a5c[1] = +2/3.
    a5c[2] = -1/12.

    a5d = np.zeros((3, 6))
    # a5d03 -------------------------------------------------------------------
    a5d[1, 0] = 0
    a5d[1, 1] = -25/12.
    a5d[1, 2] = +48/12.
    a5d[1, 3] = -36/12.
    a5d[1, 4] = +16/12.
    a5d[1, 5] = -3/12.
    # a5d12 -------------------------------------------------------------------
    a5d[2, 0] = 0
    a5d[2, 1] = -1/4.
    a5d[2, 2] = -5/6.
    a5d[2, 3] = +3/2.
    a5d[2, 4] = -1/2.
    a5d[2, 5] = +1/12.

    return a5c, a5d


def a7std():
    ''' 7 points standard coefficients '''

    a7c = np.zeros(4)
    a7c[0] = 0.
    a7c[1] = +3/4.
    a7c[2] = -3/20.
    a7c[3] = +1/60.

    a7d = np.zeros((4, 8))
    # a7d06 -------------------------------------------------------------------
    a7d[1, 0] = 0.
    a7d[1, 1] = -49/20.
    a7d[1, 2] = +6.
    a7d[1, 3] = -15/2.
    a7d[1, 4] = +20/3.
    a7d[1, 5] = -15/4.
    a7d[1, 6] = +6/5.
    a7d[1, 7] = -1/6.
    # a7d15 -------------------------------------------------------------------
    a7d[2, 0] = 0.
    a7d[2, 1] = -1/6.
    a7d[2, 2] = -77/60.
    a7d[2, 3] = +5/2.
    a7d[2, 4] = -5/3.
    a7d[2, 5] = +5/6.
    a7d[2, 6] = -1/4.
    a7d[2, 7] = +1/30.
    # a7d24 -------------------------------------------------------------------
    a7d[3, 0] = 0.
    a7d[3, 1] = +1/30.
    a7d[3, 2] = -2/5.
    a7d[3, 3] = -7/12.
    a7d[3, 4] = +4/3.
    a7d[3, 5] = -1/2.
    a7d[3, 6] = +2/15.
    a7d[3, 7] = -1/60.

    return a7c, a7d


def a7drp():
    ''' 7 pts DRP Schemes by Tam'''

    a7c = np.zeros(4)
    a7c[1] = +0.770882380518225552
    a7c[2] = -0.166705904414580469
    a7c[3] = +0.0208431427703117643
    a7c[0] = 0.

    # a7d06 -------------------------------------------------------------------
    a7d = np.zeros((4, 8))
    a7d[1, 1] = -2.192280339
    a7d[1, 2] = +4.748611401
    a7d[1, 3] = -5.108851915
    a7d[1, 4] = +4.461567104
    a7d[1, 5] = -2.833498741
    a7d[1, 6] = +1.128328861
    a7d[1, 7] = -0.203876371
    # a7d15 -------------------------------------------------------------------
    a7d[2, 1] = -0.209337622
    a7d[2, 2] = -1.084875676
    a7d[2, 3] = +2.147776050
    a7d[2, 4] = -1.388928322
    a7d[2, 5] = +0.768949766
    a7d[2, 6] = -0.281814650
    a7d[2, 7] = 0.048230454
    # a7d24 -------------------------------------------------------------------
    a7d[3, 1] = +0.049041958
    a7d[3, 2] = -0.468840357
    a7d[3, 3] = -0.474760914
    a7d[3, 4] = +1.273274737
    a7d[3, 5] = -0.518484526
    a7d[3, 6] = +0.166138533
    a7d[3, 7] = -0.026369431

    return a7c, a7d
