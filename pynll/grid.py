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
import numpy as np


valid_phases = ('P', 'S')

valid_grid_types = (
    'VELOCITY',
    'VELOCITY_METERS',
    'SLOWNESS',
    'VEL2',
    'SLOW2',
    'SLOW2_METERS',
    'SLOW_LEN',
    'STACK',
    'TIME',
    'TIME2D',
    'PROB_DENSITY',
    'MISFIT',
    'ANGLE',
    'ANGLE2D'
)

valid_float_types = {
    # NLL_type: numpy_type
    'FLOAT': 'float32',
    'DOUBLE': 'float64'
}

valid_float_types = {
    # NLL_type: numpy_type
    'FLOAT': 'float32',
    'DOUBLE': 'float64'
}

valid_grid_units = (
    'METER',
    'KILOMETER'
)


def test(ranou):
    print(ranou)
    return


class Grid3D(object):
    """
    base 3D rectilinear grid object
    """

    __valid_grid_types__ = (
        'VELOCITY',
        'VELOCITY_METERS',
        'SLOWNESS',
        'VEL2',
        'SLOW2',
        'SLOW2_METERS',
        'SLOW_LEN',
        'STACK',
        'TIME',
        'TIME2D',
        'PROB_DENSITY',
        'MISFIT',
        'ANGLE',
        'ANGLE2D'
    )

    def __init__(self, base_name, phase, origin_x, origin_y, origin_z,
                 spacing_x, spacing_y, spacing_z, n_x, n_y, n_z,
                 seed_x=None, seed_y=None, seed_z=None,
                 grid_type='VELOCITY', grid_units='METER',
                 float_type="FLOAT"):

        self.base_name = base_name

        if phase in valid_phases:
            self.phase = phase
        else:
            msg = f'phase should be one of the following valid phases:\n'
            for valid_phase in valid_phases:
                msg += f'{valid_phase}\n'
            raise ValueError(msg)

        self.origin_x = origin_x
        self.origin_y = origin_y
        self.origin_z = origin_z
        self.origin = np.array([origin_x, origin_y, origin_z])

        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.spacing_z = spacing_z

        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        if grid_type.upper() == 'TIME':
            if (not self.seed_x) or (not self.seed_y) or (not self.seed_z):
                raise ValueError('the seeds value must be set when a TIME'
                                 'is specified')

        self.array = np.zeros(n_x, n_y, n_z)

        if grid_type.upper() in valid_grid_types:
            self.grid_type = grid_type.upper()
        else:
            msg = f'grid_type should be one of the following valid grid ' \
                  f'types:\n'
            for valid_grid_type in valid_grid_types:
                msg += f'{valid_grid_type}\n'
            raise ValueError(msg)

        if grid_units.upper() in valid_grid_units:
            self.grid_units = grid_units.upper()
        else:
            msg = f'grid_units should be one of the following valid grid ' \
                  f'units:\n'
            for valid_grid_unit in valid_grid_units:
                msg += f'{valid_grid_unit}\n'
            raise ValueError(msg)

        if float_type.upper() in valid_float_types.keys():
            self.float_type = float_type
        else:
            msg = f'float_type should be one of the following valid float ' \
                  f'types:\n'
            for valid_float_type in valid_float_types:
                msg += f'{valid_float_type}\n'
            raise ValueError(msg)


    def _write_grid_data(base_name, data):
        """
        write 3D grid data to a NLLoc grid
        :param base_name: file name without the extension (.buf extension will be
        added automatically)
        :type base_name: str
        :param data: 3D grid data to be written
        :type data: 3D numpy.array
        :rtype: None
        """
    with open(base_name + '.buf', 'wb') as ofile:
        ofile.write(data.astype(np.float32).tobytes())


def _write_grid_header():
    """
    write NLLoc grid header file
    :param base_name: file name without the extension (.buf extension will be
    added automatically)
    :type base_name: str
    :param shape: grid shape
    :type shape: tuple, list or numpy.array
    :param origin: grid origin
    :type origin: tuple, list or numpy.array
    :param spacing: grid spacing
    :type spacing: float
    :param grid_type: type of NLLoc grid. For valid choice see below. Note that
    the grid_type is not case sensitive (e.g., 'velocity' == 'VELOCITY')
    :type grid_type: str
    :param station: station code or name (required only for certain grid type)
    :type station: str
    :param seed: the station location (required only for certain grid type)
    :type seed: tuple, list or numpy.array

    """

    line1 = u"%d %d %d  %f %f %f  %f %f %f  %s\n" % (
        shape[0], shape[1], shape[2],
        origin[0] / 1000., origin[1] / 1000., origin[2] / 1000.,
        spacing / 1000., spacing / 1000., spacing / 1000.,
        grid_type)

    with open(base_name + '.hdr', 'w') as ofile:
        ofile.write(line1)

        if grid_type in ['TIME', 'ANGLE']:
            line2 = u"%s %f %f %f\n" % (station, seed[0], seed[1], seed[2])
            ofile.write(line2)

        ofile.write(u'TRANSFORM  NONE\n')

    return


    def write_nll_format(self,base_name, data, origin, spacing, grid_type, seed=None,
                   label=None):
    """
    Write write structure data grid to NLLoc grid format
    :param base_name: output file name and path without extension
    :type base_name: str
    :param data: structured data
    :type data: numpy.ndarray
    :param origin: grid origin
    :type origin: tuple
    :param spacing: spacing between grid nodes (same in all dimensions)
    :type spacing: float
    :param grid_type: type of grid (must be a valid NLL grid type)
    :type grid_type: str
    :param seed: seed of the grid value. Only required / used for "TIME" or
    "ANGLE" grids
    :type seed: tuple
    :param label: seed label (usually station code). Only required / used for
    "TIME" and "ANGLE" grids
    :type label: str
    :param velocity_to_slow_len: convert "VELOCITY" to "SLOW_LEN". NLLoc
    Grid2Time program requires that "VELOCITY" be expressed in "SLOW_LEN"
    units.
    Has influence only if the grid_type is "VELOCITY"
    :type velocity_to_slow_len: bool
    :rtype: None

    supported NLLoc grid types are

    "VELOCITY": velocity (km/sec);
    "VELOCITY_METERS": velocity (m/sec);
    "SLOWNESS = slowness (sec/km);
    "SLOW_LEN" = slowness*length (sec);
    "TIME" = time (sec) 3D grid;
    "PROB_DENSITY" = probability density;
    "MISFIT" = misfit (sec);
    "ANGLE" = take-off angles 3D grid;
    """

    # removing the extension if extension is part of the base name

    if ('.buf' == base_name[-4:]) or ('.hdr' == base_name[-4:]):
        # removing the extension
        base_name = base_name[:-4]

    if (grid_type == 'VELOCITY') and (velocity_to_slow_len):
        tmp_data = spacing / data  # need this to be in SLOW_LEN format (s/km2)
        grid_type = 'SLOW_LEN'
    else:
        tmp_data = data

    _write_grid_data(base_name, tmp_data)

    shape = data.shape

    _write_grid_header(base_name, shape, origin, spacing,
                       grid_type, label, seed)

