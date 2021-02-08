# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: <filename>
#  Purpose: <purpose>
#   Author: <author>
#    Email: <email>
#
# Copyright (C) <copyright>
# --------------------------------------------------------------------
"""


:copyright:
    <copyright>
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

from nlloc.grid import NLLocGrid


def test_create_nlloc_grid():

    dims = [100, 100, 100]
    origin = [500, 1000, 10000]
    spacing = [10, 5, 10]
    phase = 'P'

    grid = NLLocGrid('test', dims, origin, spacing, phase,
                     seed=None, seed_label=None,
                     grid_type='VELOCITY_METERS', grid_units='METER',
                     float_type="FLOAT", resource_id=None)

    assert grid.write()