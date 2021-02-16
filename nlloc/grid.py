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
from uquake.core.grid import Grid
from uquake.core.grid import read_grid
from pathlib import Path
from glob import glob
from uuid import uuid4
import matplotlib.pyplot as plt


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

valid_grid_units = (
    'METER',
    'KILOMETER'
)

__velocity_grid_location__ = Path('model')
__time_grid_location__ = Path('time')


def validate_phase(phase):
    if phase not in valid_phases:
        msg = f'phase should be one of the following valid phases:\n'
        for valid_phase in valid_phases:
            msg += f'{valid_phase}\n'
        raise ValueError(msg)
    return True

def validate_grid_type(grid_type):
    if grid_type.upper() not in valid_grid_types:
        msg = f'grid_type = {grid_type} is not valid\n' \
              f'grid_type should be one of the following valid grid ' \
              f'types:\n'
        for valid_grid_type in valid_grid_types:
            msg += f'{valid_grid_type}\n'
        raise ValueError(msg)
    return True

def validate_grid_units(grid_units):
    if grid_units.upper() not in valid_grid_units:
        msg = f'grid_units = {grid_units} is not valid\n' \
              f'grid_units should be one of the following valid grid ' \
              f'units:\n'
        for valid_grid_unit in valid_grid_units:
            msg += f'{valid_grid_unit}\n'
        raise ValueError(msg)
    return True

def validate_float_type(float_type):
    if float_type.upper() not in valid_float_types.keys():
        msg = f'float_type = {float_type} is not valid\n' \
              f'float_type should be one of the following valid float ' \
              f'types:\n'
        for valid_float_type in valid_float_types:
            msg += f'{valid_float_type}\n'
        raise ValueError(msg)
    return True


class NLLocGrid(Grid):
    """
    base 3D rectilinear grid object
    """
    def __init__(self, base_name, data_or_dims, origin, spacing, phase,
                 seed=None, seed_label=None, value=0,
                 grid_type='VELOCITY_METERS', grid_units='METER',
                 float_type="FLOAT", model_id=None):
        super().__init__(data_or_dims, spacing=spacing, origin=origin,
                         value=value, resource_id=model_id)

        self.base_name = base_name

        if validate_phase(phase):
            self.phase = phase.upper()

        if validate_grid_type(grid_type):
            self.grid_type = grid_type.upper()

        if grid_type.upper() in ['TIME', 'TIME2D', 'ANGLE', 'ANGLE2D']:
            if not seed:
                raise ValueError('the seeds value must be set for TIME and '
                                 'ANGLE grids')
            if not seed_label:
                raise ValueError('the seed_label must be set for TIME '
                                 'and ANGLE grids')

        self.seed = seed
        self.seed_label = seed_label

        if validate_grid_units(grid_units):
            self.grid_units = grid_units.upper()

        if validate_float_type(float_type):
            self.float_type = float_type.upper()

    @classmethod
    def from_file(cls, base_name, path='.', float_type='FLOAT', phase='P'):
        """
        read two parts NLLoc files
        :param base_name:
        :param path: location of grid files
        :param float_type: float type as defined in NLLoc grid documentation
        """
        header_file = Path(path) / f'{base_name}.hdr'
        with open(header_file, 'r') as in_file:
            line = in_file.readline()
            line = line.split()
            shape = tuple([int(line[0]), int(line[1]), int(line[2])])
            origin = np.array([float(line[3]), float(line[4]),
                                     float(line[5])]) * 1000
            # dict_out.origin *= 1000
            spacing = np.array([float(line[6]), float(line[7]),
                                float(line[8])]) * 1000

            grid_type = line[9]
            grid_unit = 'METER'

            line = in_file.readline()

            if grid_type in ['ANGLE', 'ANGLE2D', 'TIME', 'TIME2D']:
                line = line.split()
                seed_label = line[0]
                seed = (float(line[1]) * 1000,
                        float(line[2]) * 1000,
                        float(line[3]) * 1000)
            else:
                seed_label = None
                seed = None

        buf_file = Path(path) / f'{base_name}.buf'
        if float_type == 'FLOAT':
            data = np.fromfile(buf_file,
                               dtype=np.float32)
        elif float_type == 'DOUBLE':
            data = np.fromfile(buf_file,
                               dtype=np.float64)
        else:
            msg = f'float_type = {float_type} is not valid\n' \
                  f'float_type should be one of the following valid float ' \
                  f'types:\n'
            for valid_float_type in valid_float_types:
                msg += f'{valid_float_type}\n'
            raise ValueError(msg)

        data = data.reshape(shape)

        if '.P.' in base_name:
            phase = 'P'
        else:
            phase = 'S'

        # reading the model id file
        mid_file = Path(path) / f'{base_name}.mid'
        if mid_file.exists():
            with open(mid_file, 'r') as mf:
                model_id = mf.readline().strip()

        else:
            model_id = str(uuid4())

            # (self, base_name, data_or_dims, origin, spacing, phase,
            #  seed=None, seed_label=None, value=0,
            #  grid_type='VELOCITY_METERS', grid_units='METER',
            #  float_type="FLOAT", model_id=None):

        return cls(base_name, data, origin, spacing, phase, seed=seed,
                   seed_label=seed_label, grid_type=grid_type,
                   model_id=model_id)

    def _write_grid_data(self, path='.'):

        with open(Path(path) / (self.base_name + '.buf'), 'wb') \
                as out_file:
            if self.float_type == 'FLOAT':
                out_file.write(self.data.astype(np.float32).tobytes())

            elif self.float_type == 'DOUBLE':
                out_file.write(self.data.astype(np.float64).tobytes())

    def _write_grid_header(self, path='.'):

        # convert 'METER' to 'KILOMETER'
        if self.grid_units == 'METER':
            origin = self.origin / 1000
            spacing = self.spacing / 1000

        line1 = f'{self.shape[0]:d} {self.shape[1]:d} {self.shape[1]:d}  ' \
                f'{origin[0]:f} {origin[1]:f} {origin[2]:f}  ' \
                f'{spacing[0]:f} {spacing[1]:f} {spacing[2]:f}  ' \
                f'{self.grid_type}\n'

        with open(Path(path) / (self.base_name + '.hdr'), 'w') as out_file:
            out_file.write(line1)

            if self.grid_type in ['TIME', 'ANGLE']:

                if self.grid_units == 'METER':
                    seed = self.seed / 1000

                line2 = u"%s %f %f %f\n" % (self.seed_label,
                                            seed[0], seed[1], seed[2])
                out_file.write(line2)

            out_file.write(u'TRANSFORM  NONE\n')

        return True

    def _write_grid_model_id(self, path='.'):
        with open(Path(path) / (self.base_name + '.mid'), 'w') as out_file:
            out_file.write(f'{self.model_id}')
        return True

    def write(self, path='.'):

        # removing the extension if extension is part of the base name

        if ('.buf' == self.base_name[-4:]) or ('.hdr' == self.base_name[-4:]):
            # removing the extension
            self.base_name = self.base_name[:-4]

        # if (grid_type == 'VELOCITY') and (velocity_to_slow_len):
        #     tmp_data = spacing / data  # need this to be in SLOW_LEN format
        #     (s/km2)
        #     grid_type = 'SLOW_LEN'
        # else:
        #     tmp_data = data

        self._write_grid_data(path=path)
        self._write_grid_header(path=path)
        self._write_grid_model_id(path=path)

        return True

    @property
    def model_id(self):
        return self.resource_id

    @property
    def sensor(self):
        return self.seed_label


class ModelLayer:
    """
    1D model varying in Z
    """

    def __init__(self, z_top, value_top):
        """
        :param z_top: Top of the layer z coordinates
        :param value_top: Value at the top of the layer
        """
        self.z_top = z_top
        self.value_top = value_top


class LayeredVelocityModel:

    def __init__(self, project_code, model_id=None, velocity_model_layers=[],
                 phase='P', grid_units='METER', float_type='FLOAT'):
        """
        Initialize
        :param model_id: model id, if not set the model ID is set using UUID
        :param velocity_model_layers: a list of VelocityModelLayer
        :param phase:
        """
        self.project_code = project_code
        self.layers = velocity_model_layers

        if validate_phase(phase):
            self.phase = phase.upper()

        if validate_grid_units(grid_units):
            self.grid_units = grid_units.upper()

        if validate_float_type(float_type):
            self.float_type = float_type.upper()

        self.grid_type = 'VELOCITY'

        if model_id is None:
            model_id = str(uuid4())

        self.model_id = model_id

    def add_layer(self, layer):
        """
        Add a layer to the model. The layers must be added in sequence from the
        top to the bottom
        :param layer: a LayeredModel object
        :return: None
        """
        if not (type(layer) is ModelLayer):
            raise TypeError('layer must be a VelocityModelLayer object')

        self.layers.append(layer)

    def gen_1d_model(self, z_min, z_max, spacing):
        # sort the layers to ensure the layers are properly ordered
        z = []
        v = []
        for layer in self.layers:
            z.append(layer.z_top)
            v.append(layer.value_top)

        if np.max(z) < z_max:
            i_z_max = np.argmax(z)
            v_z_max = v[i_z_max]

            z.append(z_max)
            v.append(v_z_max)

        if np.min(z) > z_min:
            i_z_min = np.argmin(z)
            v_z_min = v[i_z_min]

            z.append(z_min)
            v.append(v_z_min)

        i_sort = np.argsort(z)

        z = np.array(z)
        v = np.array(v)

        z = z[i_sort]
        v = v[i_sort]

        z_interp = np.arange(z_min, z_max, spacing)
        v_interp = np.interp(z_interp, z, v)

        return z_interp, v_interp

    def gen_3d_grid(self, dims, origin, spacing):
        model_grid_3d = VelocityGrid3D.from_layered_model(self, dims, origin,
                                                          spacing)
        return model_grid_3d

    def plot(self, z_min, z_max, spacing, *args, **kwargs):
        """
        Plot the 1D velocity model
        :param z_min: lower limit of the model
        :param z_max: upper limit of the model
        :param spacing: plotting resolution in z
        :return: matplotlib axis
        """

        z_interp, v_interp = self.gen_1d_model(z_min, z_max, spacing)

        if self.grid_type == 'Vp':
            x_label = 'P-wave velocity'
        elif self.grid_type == 'Vs':
            x_label = 's-wave velocity'
        elif self.grid_type == 'rho':
            x_label = 'density'

        if self.units == 'METER':
            units = 'm'
        else:
            units = 'km'

        y_label = f'z [{units}]'
        ax = plt.axes()
        ax.plot(v_interp, z_interp, *args, **kwargs)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        ax.set_aspect(2)

        plt.tight_layout()

        return ax


class VelocityGrid3D(NLLocGrid):

    def __init__(self, project_code, data_or_dims, origin, spacing,
                 phase='P', path='.', value=0, float_type='FLOAT',
                 model_id=None, **kwargs):
        self.project_code = project_code
        self.path = path
        base_name = f'{project_code}.{phase.upper()}.mod'
        super().__init__(base_name, data_or_dims, origin, spacing, phase,
                 seed=None, seed_label=None, value=value,
                 grid_type='VELOCITY_METERS', grid_units='METER',
                 float_type=float_type, model_id=model_id)

    @classmethod
    def from_layered_model(cls, layered_model, dims, origin,
                           spacing, **kwargs):
        """
        Generating a 3D grid model from
        :param layered_model:
        :param code:
        :param dims:
        :param origin:
        :param spacing:
        :param kwargs:
        :return:
        """

        z_min = origin[-1]
        z_max = z_min + spacing[-1] * dims[-1]

        z_interp, v_interp = layered_model.gen_1d_model(z_min, z_max,
                                                        spacing[-1])

        data = np.zeros(dims)

        for i, v in enumerate(v_interp):
            data[:, :, i] = v_interp[i]

        return cls(layered_model.project_code, data, origin, spacing,
                   phase=layered_model.phase,
                   float_type=layered_model.float_type,
                   model_id=layered_model.model_id, **kwargs)

    def gen_derivative_grids(self, time=True, angle=True, take_off_angle=True):
        """
        Generate the travel-time and
        :param time:
        :param angle:
        :param take_off_angle:
        :return:
        """
        pass





    def add_layer(self):
        pass

