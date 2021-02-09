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

dims = [100, 100, 100]
origin = [500, 1000, 10000]
spacing = [10, 5, 10]
phase = 'P'

base_name = 'test'

grid = NLLocGrid(base_name, dims, origin, spacing, phase,
                 seed=None, seed_label=None,
                 grid_type='VELOCITY_METERS', grid_units='METER',
                 float_type="FLOAT", model_id=None)


def test_create_nlloc_grid():

    assert grid.write()


def test_read_nlloc_grid():

    new_grid = NLLocGrid.from_file(base_name, path='.', float_type='FLOAT')

    assert new_grid == grid