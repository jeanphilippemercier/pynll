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
import matplotlib.pyplot as plt

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


def test_layered_model():

    from nlloc.grid import (ModelLayer, LayeredVelocityModel)

    # The origin is the lower left corner
    project_code = 'test'
    origin = [650200, 4766170, -500]
    dimensions = [100, 101, 68]
    spacing = [25, 25, 25]

    z = [1168, 459, -300, -500]
    vp_z = [4533, 5337, 5836, 5836]
    vs_z = [2306, 2885, 3524, 3524]

    layered_model = LayeredVelocityModel(project_code)
    for (z_, vp) in zip(z, vp_z):
        layer = ModelLayer(z_, vp)
        layered_model.add_layer(layer)

    z_bottom = origin[2]
    z_top = origin[2] + spacing[2] * dimensions[2]
    z_interp, v_interp = layered_model.gen_1d_model(z_bottom, z_top,
                                                    spacing[2])

    grid_3d = layered_model.gen_3d_grid(dimensions, origin, spacing)

    assert True



