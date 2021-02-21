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
from pathlib import Path
from uuid import uuid4
import matplotlib.pyplot as plt
from loguru import logger
import skfmm


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
    'KILOMETER',
    'SECOND',
    'DEGREE'
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
    def __init__(self, data_or_dims, origin, spacing, phase,
                 value=0, grid_type='VELOCITY_METERS', grid_units='METER',
                 float_type="FLOAT", model_id=None):
        """

        :param base_name: file base name
        :type base_name: str
        :param data_or_dims: data or data dimensions. If dimensions are
        provided the a homogeneous gris is created with value=value
        :param origin: origin of the grid
        :type origin: list
        :param spacing: the spacing between grid nodes
        :type spacing: list
        :param phase: the seismic phase (value 'P' or 'S')
        :type phase: str
        :param seed: seed of the grid (for travel-time grid only)
        :type seed: numpy.array
        :param seed_label:
        :param value:
        :param grid_type:
        :param grid_units:
        :param float_type:
        :param model_id:
        """

        super().__init__(data_or_dims, spacing=spacing, origin=origin,
                         value=value, resource_id=model_id)

        if validate_phase(phase):
            self.phase = phase.upper()

        if validate_grid_type(grid_type):
            self.grid_type = grid_type.upper()

        # if grid_type.upper() in ['TIME', 'TIME2D', 'ANGLE', 'ANGLE2D']:
        #     if not seed:
        #         raise ValueError('the seeds value must be set for TIME and '
        #                          'ANGLE grids')
        #     if not seed_label:
        #         raise ValueError('the seed_label must be set for TIME '
        #                          'and ANGLE grids')

        # self.seed = seed
        # self.seed_label = seed_label

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

    def _write_grid_data(self, base_name, path='.'):

        Path(path).mkdir(parents=True, exist_ok=True)

        with open(Path(path) / (base_name + '.buf'), 'wb') \
                as out_file:
            if self.float_type == 'FLOAT':
                out_file.write(self.data.astype(np.float32).tobytes())

            elif self.float_type == 'DOUBLE':
                out_file.write(self.data.astype(np.float64).tobytes())

    def _write_grid_header(self, base_name, path='.', seed_label=None,
                           seed=None, seed_units=None):

        # convert 'METER' to 'KILOMETER'
        if self.grid_units == 'METER':
            origin = self.origin / 1000
            spacing = self.spacing / 1000

        line1 = f'{self.shape[0]:d} {self.shape[1]:d} {self.shape[2]:d}  ' \
                f'{origin[0]:f} {origin[1]:f} {origin[2]:f}  ' \
                f'{spacing[0]:f} {spacing[1]:f} {spacing[2]:f}  ' \
                f'{self.grid_type}\n'

        with open(Path(path) / (base_name + '.hdr'), 'w') as out_file:
            out_file.write(line1)
            out_file.write(f'{self.model_id}')

            if self.grid_type in ['TIME', 'ANGLE']:

                if seed_units is None:
                    logger.warning(f'seed_units are not defined. '
                                   f'Assuming same units as grid ('
                                   f'{self.grid_units}')
                if self.grid_units == 'METER':
                    seed = seed / 1000

                line2 = u"%s %f %f %f\n" % (seed_label,
                                            seed[0], seed[1], seed[2])
                out_file.write(line2)

            out_file.write(u'TRANSFORM  NONE\n')

        return True

    def _write_grid_model_id(self, base_name, path='.'):
        with open(Path(path) / (base_name + '.mid'), 'w') as out_file:
            out_file.write(f'{self.model_id}')
        return True

    def write(self, base_name, path='.', seed_label=None, seed=None,
              seed_units=None):
        """

        """

        # removing the extension if extension is part of the base name

        if self.grid_type in ['TIME', 'TIME2D', 'ANGLE', 'ANGLE2D']:
            if (seed_label is None) | (seed is None):
                raise ValueError(f'the seed_label and seed parameters must be'
                                 f'specified for grid_type in '
                                 f'{self.grid_type}')

        self._write_grid_data(base_name, path=path)
        self._write_grid_header(base_name, path=path, seed_label=seed_label,
                                seed=seed)
        self._write_grid_model_id(base_name, path=path)

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

    def __init__(self, model_id=None, velocity_model_layers=[],
                 phase='P', grid_units='METER', float_type='FLOAT'):
        """
        Initialize
        :param model_id: model id, if not set the model ID is set using UUID
        :type model_id: str
        :param velocity_model_layers: a list of VelocityModelLayer
        :type velocity_model_layers: list
        :param phase: Phase either 'P' or 'S'
        :type phase: str
        """

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

        z_interp = np.arange(z_min, z_max, spacing[2])
        v_interp = np.interp(z_interp, z, v)

        return z_interp, v_interp

    def gen_3d_grid(self, network_code, dims, origin, spacing):
        model_grid_3d = VelocityGrid3D.from_layered_model(self,
                                                          network_code,
                                                          dims, origin,
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

    def __init__(self, network_code, data_or_dims, origin, spacing,
                 phase='P', value=0, float_type='FLOAT',
                 model_id=None, **kwargs):

        self.network_code = network_code

        if (type(spacing) is int) | (type(spacing) is float):
            spacing = [spacing, spacing, spacing]

        super().__init__(data_or_dims, origin, spacing, phase,
                         value=value, grid_type='VELOCITY_METERS',
                         grid_units='METER', float_type=float_type,
                         model_id=model_id)

    @classmethod
    def from_layered_model(cls, layered_model, network_code, dims, origin,
                           spacing, **kwargs):
        """
        Generating a 3D grid model from
        :param network_code:
        :param layered_model:
        :param dims:
        :param origin:
        :param spacing:
        :param kwargs:
        :return:
        """

        z_min = origin[-1]
        z_max = z_min + spacing[-1] * dims[-1]

        z_interp, v_interp = layered_model.gen_1d_model(z_min, z_max,
                                                        spacing)

        data = np.zeros(dims)

        for i, v in enumerate(v_interp):
            data[:, :, i] = v_interp[i]

        return cls(network_code, data, origin, spacing,
                   phase=layered_model.phase,
                   float_type=layered_model.float_type,
                   model_id=layered_model.model_id, **kwargs)

    def to_slow_lens(self):
        data = self.spacing[0] / self.data

        return NLLocGrid(data, self.origin, self.spacing,
                         self.phase, grid_type='SLOW_LEN',
                         grid_units=self.grid_units,
                         float_type=self.float_type,
                         model_id=self.model_id)

    def to_time(self, seed, seed_label, *args, **kwargs):
        """
        Eikonal solver based on scikit fast marching solver
        :param seed: numpy array location of the seed or origin of seismic wave
         in model coordinates
        (usually location of a station or an event)
        :type seed: numpy array
        :param seed_label: seed label (name of station)
        :type seed_label: basestring
        :rtype: TTGrid
        """

        if not self.in_grid(seed):
            logger.warning(f'{seed_label} is outside the grid. '
                           f'The travel time grid will not be calculated')
            return

        seed = np.array(seed)

        phi = -1 * np.ones_like(self.data)
        seed_coord = self.transform_to(seed)

        phi[tuple(seed_coord.astype(int))] = 1

        tt = skfmm.travel_time(phi, self.data, dx=self.spacing, *args,
                               **kwargs)

        return TTGrid(self.network_code, tt, self.origin, self.spacing,
                      seed, seed_label, phase=self.phase,
                      float_type=self.float_type, model_id=self.model_id)

    def to_time_multi_threaded(self, seeds, seed_labels, *args, **kwargs):
        """
        Multithreaded version of the Eikonal solver
        based on scikit fast marching solver
        :param seeds: array of seed
        :type seeds: np.array
        :param seed_labels: array of seed_labels
        :type seed_labels: np.array
        :param args: arguments to be passed directly to skfmm.travel_time
        function
        :param kwargs: keyword arguments to be passed directly to
        skfmm.travel_time function
        :return: a list of TTGrids
        """

        from multiprocessing import Pool, cpu_count
        import contextlib

        num_threads = int(np.ceil(0.95 * cpu_count()))

        data = []
        for seed, seed_label in zip(seeds, seed_labels):
            if not self.in_grid(seed):
                logger.warning(f'{seed_label} is outside the grid. '
                               f'The travel time grid will not be calculated')
                continue
            data.append((seed, seed_label))

        with Pool(num_threads) as pool:
            results = pool.starmap(self.to_time, data)

        return results

    def write(self, path='.'):

        base_name = self.base_name
        super().write(base_name, path=path)

    @property
    def base_name(self):
        return f'{self.network_code.upper()}.{self.phase.upper()}.mod'


class SeededGrid(NLLocGrid):
    """
    container for seeded grids (e.g., travel time, azimuth and take off angle)
    """
    __valid_grid_type__ = ['TIME', 'TIME2D', 'ANGLE', 'ANGLE2D']

    def __init__(self, network_code, data_or_dims, origin, spacing, seed,
                 seed_label, phase='P', value=0,
                 grid_type='TIME', float_type="FLOAT", model_id=None):
        self.seed = seed
        self.seed_label = seed_label
        self.network_code = network_code

        if grid_type not in self.__valid_grid_type__:
            raise ValueError()
        self.grid_type = grid_type

        super().__init__(data_or_dims, origin, spacing,
                         phase=phase, value=value,
                         grid_type='TIME', grid_units='SECOND',
                         float_type=float_type, model_id=model_id)

    @property
    def base_name(self):
        base_name = f'{self.network_code}.{self.phase}.{self.seed_label}.' \
                    f'{self.grid_type.lower()}'
        return base_name


class TTGrid(SeededGrid):
    def __init__(self, network_code, data_or_dims, origin, spacing, seed,
                 seed_label, phase='P', value=0, float_type="FLOAT",
                 model_id=None):

        super().__init__(network_code, data_or_dims, origin, spacing, seed,
                         seed_label, phase=phase, value=value,
                         grid_type='TIME', float_type=float_type,
                         model_id=model_id)

    def to_azimuth(self):
        """
        This function calculate the take off angle and azimuth for every grid point
        given a travel time grid calculated using an Eikonal solver
        :param travel_time: travel_time grid
        :type travel_time: ~microquake.core.data.grid.GridData with seed property
        (travel_time.seed).
        :rparam: azimuth and takeoff angles grids
        .. Note: The convention for the takeoff angle is that 0 degree is down.
        """

        gds_tmp = np.gradient(self.data)
        gds = [-gd for gd in gds_tmp]

        azimuth = np.arctan2(gds[0], gds[1])  # azimuth is zero northwards

        return AngleGrid(self.network_code, azimuth, self.origin, self.spacing,
                         self.seed, self.seed_label, 'AZIMUTH',
                         phase=self.phase, float_type=self.float_type,
                         model_id=self.model_id)

    def to_takeoff(self):
        gds_tmp = np.gradient(self.data)
        gds = [-gd for gd in gds_tmp]

        hor = np.sqrt(gds[0] ** 2 + gds[1] ** 2)
        takeoff = np.arctan2(hor, -gds[2])
        # takeoff is zero pointing down
        return AngleGrid(self.network_code, takeoff, self.origin, self.spacing,
                         self.seed, self.seed_label, 'TAKEOFF',
                         phase=self.phase, float_type=self.float_type,
                         model_id=self.model_id)

    def to_azimuth_point(self, coord, grid_coordinates=False, mode='linear',
                         order=1, **kwargs):
        """
        calculate the azimuth angle at a particular point on the grid for a
        given seed location
        :param coord: coordinates at which to calculate the takeoff angle
        :param grid_coordinates: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param mode: interpolation mode
        :param order: interpolation order
        :return: takeoff angle at the location coord
        """

        return self.to_azimuth().interpolate(coord,
                                             grid_coordinates=grid_coordinates,
                                             mode=mode, order=order, **kwargs)

    def to_takeoff_point(self, coord, grid_coordinates=False, mode='linear',
                         order=1, **kwargs):
        """
        calculate the takeoff angle at a particular point on the grid for a
        given seed location
        :param coord: coordinates at which to calculate the takeoff angle
        :param grid_coordinates: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param mode: interpolation mode
        :param order: interpolation order
        :return: takeoff angle at the location coord
        """
        return self.to_takeoff().interpolate(coord,
                                             grid_coordinates=model_space,
                                             mode=mode, order=order, **kwargs)

    def ray_tracer(self, start, grid_coordinates=False, max_iter=1000,
                   arrival_id=None):
        """
        This function calculates the ray between a starting point (start) and an
        end point, which should be the seed of the travel_time grid, using the
        gradient descent method.
        :param start: the starting point (usually event location)
        :type start: tuple, list or numpy.array
        :param grid_coordinates: true if the coordinates are expressed in
        grid space (indices can be fractional) as opposed to model space
        (x, y, z)
        :param max_iter: maximum number of iteration
        :param arrival_id: id of the arrival associated to the ray if
        applicable
        :rtype: numpy.array
        """

        from uquake.core.event import Ray

        if grid_coordinates:
            start = np.array(start)
            start = self.transform_from(start)

        origin = self.origin
        spacing = self.spacing
        end = np.array(self.seed)
        start = np.array(start)

        # calculating the gradient in every dimension at every grid points
        gds = [Grid(gd, origin=origin, spacing=spacing)
               for gd in np.gradient(self.data)]

        dist = np.linalg.norm(start - end)
        cloc = start  # initializing cloc "current location" to start
        gamma = spacing / 2  # gamma is set to half the grid spacing. This
        # should be
        # sufficient. Note that gamma is fixed to reduce
        # processing time.
        nodes = [start]

        iter_number = 0
        while np.all(dist > spacing / 2):
            if iter_number > max_iter:
                break

            if np.all(dist < spacing * 4):
                gamma = np.min(spacing) / 4

            gvect = np.array([gd.interpolate(cloc, grid_coordinate=False,
                                             order=1)[0] for gd in gds])

            cloc = cloc - gamma * gvect / np.linalg.norm(gvect)
            nodes.append(cloc)
            dist = np.linalg.norm(cloc - end)

            iter_number += 1

        nodes.append(end)

        ray = Ray(nodes=nodes, station_code=self.seed_label,
                  phase=self.phase)

        return ray

    @classmethod
    def from_velocity(cls, seed, seed_label, velocity_grid):
        return velocity_grid.eikonal(seed, seed_label)


class AngleGrid(SeededGrid):
    def __init__(self, network_code, data_or_dims, origin, spacing, seed,
                 seed_label, angle_type, phase='P', value=0, float_type="FLOAT",
                 model_id=None):

        self.angle_type = angle_type
        super().__init__(network_code, data_or_dims, origin, spacing, seed,
                         seed_label, phase=phase, value=value,
                         grid_type='ANGLE', float_type=float_type,
                         model_id=model_id)

