# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: core.py
#  Purpose: module to interact with the NLLoc
#   Author: uquake development team
#    Email: devs@uquake.org
#
# Copyright (C) 2016 uquake development team
# --------------------------------------------------------------------
"""
module to interact with the NLLoc

:copyright:
    uquake development team (devs@uquake.org)
:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
"""

import os
import shutil
import tempfile
from datetime import datetime
from glob import glob
from struct import unpack
from time import time
import numpy as np

from obspy import UTCDateTime
from uquake.core.inventory import Inventory, read_inventory
import uquake.core.event
from uquake.core.logging import logger
from uquake.core.grid import read_grid
from uquake.core.event import (Arrival, Catalog, Origin, Event, AttribDict)

from . import grid

from uuid import uuid4
from pathlib import Path


def validate(value, choices):
    if value not in choices:
        msg = f'value should be one of the following choices\n:'
        for choice in choices:
            msg += f'{choice}\n'
        raise ValueError(msg)
    return True


def validate_type(obj, expected_type):
    if type(obj) is not expected_type:
        raise TypeError('object is not the right type')


__valid_geographic_transformation__ = ['GLOBAL', 'SIMPLE', 'NONE', 'SDC',
                                       'LAMBERT']

__valid_reference_ellipsoid__ = ['WGS-84', 'GRS-80', 'WGS-72', 'Australian',
                                 'Krasovsky', 'International', 'Hayford-1909'
                                 'Clarke-1880', 'Clarke-1866', 'Airy Bessel',
                                 'Hayford-1830', 'Sphere']


class Grid2Time:
    def __init__(self, inventory, grid_transform, base_directory, base_name,
                 verbosity=1, random_seed=1000, p_wave=True, s_wave=True,
                 calculate_angles=True, model_directory='models',
                 grid_directory='grids'):
        """
        Build the control file and run the Grid2Time program.

        Note that at this time the class does not support any geographic
        transformation

        :param inventory: inventory file
        :type inventory: uquake.core.inventory.Inventory
        :param base_directory: the base directory of the project
        :type base_directory: str
        :param base_name: the network code
        :type base_name: str
        :param verbosity: sets the verbosity level for messages printed to
        the terminal ( -1 = completely silent, 0 = error messages only,
        1 = 0 + higher-level warning and progress messages,
        2 and higher = 1 + lower-level warning and progress messages +
        information messages, ...) default: 1
        :type verbosity: int
        :param random_seed:  integer seed value for generating random number
        sequences (used by program NLLoc to generate Metropolis samples and
        by program Time2EQ to generate noisy time picks)
        :param p_wave: if True calculate the grids for the P-wave
        :type p_wave: bool
        :param s_wave: if True calculate the grids for the S-wave
        :type s_wave: bool
        :param calculate_angles: if true calculate the azimuth and the
        take-off angles
        :type calculate_angles: bool
        :param model_directory: location of the model directory relative to
        the base_directory
        :type model_directory: str
        :param grid_directory: location of the grid directory relative to
        the base_directory
        """

        self.verbosity = verbosity
        self.random_seed = random_seed
        self.base_name = base_name

        self.base_directory = Path(base_directory)
        self.velocity_model_path = self.base_directory / f'{model_directory}'
        self.grid_path = self.base_directory / f'{grid_directory}'

        # create the velocity_model_path and the grid_path if they do not exist
        self.grid_path.mkdir(parents=True, exist_ok=True)
        self.velocity_model_path.mkdir(parents=True, exist_ok=True)

        self.calculate_p_wave = p_wave
        self.calculate_s_wave = s_wave
        self.calculate_angles = calculate_angles

        if type(inventory) is not Inventory:
            raise TypeError('inventory must be '
                            '"uquake.core.inventory.Inventory" type')
        self.inventory = inventory

    def run(self):
        if self.calculate_p_wave:
            self.__write_control_file__('P')
            # run

        if self.calculate_s_wave:
            self.__write_control_file__('S')

    def __write_control_file__(self, phase):

        ctrl_dir = f'{self.base_directory}/run/'
        Path(ctrl_dir).mkdir(parents=True, exist_ok=True)

        # create the directory if the directory does not exist

        ctrl_file = ctrl_dir + f'{str(uuid4())}.ctl'
        with open(ctrl_file, 'w') as ctrl:
            # writing the control section
            ctrl.write(f'CONTROL {self.verbosity} {self.random_seed}\n')

            # writing the geographic transformation section
            ctrl.write('TRANS NONE\n')

            # writing the Grid2Time section
            out_line = f'GTFILES ' \
                       f'{self.velocity_model_path}/{self.base_name} ' \
                       f'{self.grid_path}/{self.base_name} ' \
                       f'{phase} 0\n'

            ctrl.write(out_line)

            if self.calculate_angles:
                angle_mode = 'ANGLES_YES'
            else:
                angle_mode = 'ANGLE_NO'

            ctrl.write(f'GTMODE GRID3D {angle_mode}\n')

            for sensor in self.inventory.sensors:
                # test if sensor name is shorter than 6 characters

                out_line = f'GTSRCE {sensor.code} XYZ ' \
                           f'{sensor.x / 1000:>10.6f} ' \
                           f'{sensor.y / 1000 :>10.6f} ' \
                           f'{sensor.z / 1000 :>10.6f} ' \
                           f'0.00\n'

                ctrl.write(out_line)

            ctrl.write(f'GT_PLFD 1.0e-4 {self.verbosity + 1}\n')


class NonLinLoc:
    def __init__(self, control, geographic_transformation, sensors,
                 location_input_files, search_type,
                 location_method, gaussian_model_error, search_grid,
                 signature=None, comment=None, location_output_file_type=None,
                 travel_time_model_error=None,
                 phase_identifier_mapping=None,
                 quality_to_error_mapping=None,
                 phase_statistics_parameters=None,
                 takeoff_angle_parameters=None,
                 magnitude_calculation_method=None,
                 magnitude_calculation_component=None,
                 sensor_alias_code=None, exclude_observations=None,
                 phase_time_delay=None,
                 vertical_ray_elevation_correction=None,
                 topography_mask=None, sensor_distribution_weighting=None):

        """
        create an object to run NonLinLoc
        :param control: CONTROL - Sets various general program control
        parameters.
        :type control: Control
        :param geographic_transformation: TRANS - Sets geographic to working
        coordinates transformation parameters. The GLOBAL option sets
        spherical regional/teleseismic mode, with no geographic transformation
        - most position values are input and used directly as latitude and
        longitude in degrees. The SIMPLE, SDC and LAMBERT options make
        transformations of geographic coordinates into a Cartesian/rectangular
        system. The NONE transformation performs no geographic conversion.
        :type geographic_transformation: GeographicTransformation
        :param sensors: GTSRCE - Source Description: Specifies a source
        location. One time grid and one angles grid (if requested) will
        be generated for each source. Four formats are supported:
        XYZ (rectangular grid coordinates),
        LATLON (decimal degrees for latitude/longitude),
        LATLONDM (degrees + decimal minutes for latitude/longitude) and
        LATLONDS (degrees + minutes + decimal seconds for latitude/longitude).
        :param location_input_files: LOCFILES - Input and Output File Root Name
        Specifies the directory path and filename for the phase/observation
        files, and the file root names (no extension) for the input time grids
        and the output files.
        :param search_type: LOCSEARCH - Search Type: Specifies the search type
        and search parameters. The possible search types are
        GRID (grid search), MET (Metropolis), and OCT (Octtree).
        :param location_method: LOCMETH - Location Method: Specifies the
        location method (algorithm) and method parameters.
        :param gaussian_model_error: LOCGAU - Gaussian Model Errors:
        Specifies parameters for Gaussian modelisation-error covariances
        Covariance ij between stations i and j using the relation
        ( Tarantola and Valette, 1982 ):
        Covariance ij = SigmaTime 2 exp(-0.5(Dist 2 ij )/ CorrLen 2 )
        where Dist is the distance in km between stations i and j .
        :param search_grid: LOCGRID - Search Grid Description: Specifies the
        size and other parameters of an initial or nested 3D search grid.
        The order of LOCGRID statements is critical (see Notes).

        -- optional parameters

        :param signature: LOCSIG - Signature text
        :param comment: LOCCOM - Comment text
        :param location_output_file_type: LOCHYPOUT - Output File Types:
        Specifies the filetypes to be used for output.
        :param travel_time_model_error: LOCGAU2 - Travel-Time Dependent Model
        Errors: Specifies parameters for travel-time dependent
        modelisation-error. Sets the travel-time error in proportion to the
        travel-time, thus giving effectively a station-distance weighting,
        which was not included in the standard Tarantola and Valette
        formulation used by LOCGAU. This is important with velocity model
        errors, because nearby stations would usually have less absolute error
        than very far stations, and in general it is probably more correct
        that travel-time error is a percentage of the travel-time. Preliminary
        results using LOCGAU2 indicate that this way of setting travel-time
        errors gives visible improvement in hypocenter clustering.
        (can currently only be used with the EDT location methods)
        :param phase_identifier_mapping: LOCPHASEID - Phase Identifier Mapping:
        Specifies the mapping of phase codes in the phase/observation file
        ( i.e. pg or Sn ) to standardized phase codes ( i.e. P or S ).
        :param quality_to_error_mapping: LOCQUAL2ERR - Quality to Error Mapping
        required, non-repeatable, for phase/observation file formats that do
        not include time uncertainties ; ignored, non-repeatable, otherwise:
        Specifies the mapping of phase pick qualities phase/observation file
        (i.e. 0,1,2,3 or 4) to time uncertainties in seconds
        (i.e. 0.01 or 0.5).
        :param phase_statistics_parameters: LOCPHSTAT - Phase Statistics
        parameters: Specifies selection criteria for phase residuals to be
        included in calculation of average P and S station residuals.
        The average residuals are saved to a summary, phase statistics file
        (see Phase Statistics file formats ).
        :param takeoff_angle_parameters: LOCANGLES - Take-off Angles parameters
        Specifies whether to determine take-off angles for the maximum
        likelihood hypocenter and sets minimum quality cutoff for saving angles
        and corresponding phases to the HypoInverse Archive file.
        :param magnitude_calculation_method: LOCMAG - Magnitude Calculation
        Method: Specifies the magnitude calculation type and parameters.
        The possible magnitude types are:
        - ML_HB (Local (Richter) magnitudeMLfromHutton and Boore (1987)),
          ML = log(A f) +nlog(r/100) +K(r-100) + 3.0 +S,
        - MD_FMAG (Duration magnitudeMLfromLahr, J.C., (1989) HYPOELLIPSE),
          MD = C1 + C2log(Fc) + C3r + C4z + C5[log(Fc))2,
        :param magnitude_calculation_component: LOCCMP - Magnitude Calculation
        Component
        :param sensor_alias_code: LOCALIAS - Station Code Alias
        Specifies (1) an alias (mapping) of station codes, and (2) start and
        end dates of validity of the alias. Allows (1) station codes that vary
        over time or in different pick files to be homogenized to match codes
        in time grid files, and (2) allows selection of station data by time.
        :param exclude_observations: LOCEXCLUDE - Exclude Observations
        :param phase_time_delay: LOCDELAY - Phase Time Delays
        Specifies P and S delays (station corrections) to be subtracted from
        observed P and S times.
        :param vertical_ray_elevation_correction: LOCELEVCORR -
        Simple, vertical ray elevation correction:
        Calculates a simple elevation correction using the travel-time of a
        vertical ray from elev 0 to the elevation of the station. This control
        statement is mean to be used in GLOBAL mode with TauP or other time
        grids which use elevation 0 for the station elevation.
        :param topography_mask: LOCTOPO_SURFACE - Topographic mask for location
        search region:
        Uses a topographic surface file in GMT ascii GRD format to mask prior
        search volume to the half-space below the topography.
        :param sensor_distribution_weighting: LOCSTAWT - Station distribution
        weighting
        Calculates a weight for each station that is a function of the average
        distance between all stations used for location. This helps to correct
        for irregular station distribution, i.e. a high density of stations in
        regions such as Europe and North America and few or no stations in
        regions such as oceans. The relative weight for station i is:
        wieghti = 1.0 / [ SUMj exp(-dist2/cutoffDist2 ] where j is a station
        used for location and dist is the epicentral/great-circle distance
        between stations i and j.
        """

        if isinstance(control, Control):
            raise TypeError('control must be a type Control object ')
        self.control = control

        if ((not isinstance(geographic_transformation,
                            GeographicTransformation)) |
            (not issubclass(geographic_transformation,
                            GeographicTransformation))):
            raise TypeError('geographic_transformation must be a class or '
                            'subclass type GeographicTransformation')
        self.geographic_transformation = geographic_transformation

        if not isinstance(sensors, Sensors):
            raise TypeError('sensors must be a type Sensors object')
        self.sensors = sensors
        self.location_input_files = location_input_files
        self.location_output_files_type = location_output_file_type
        self.search_type = search_type
        self.location_method = location_method
        self.gaussian_error_model = gaussian_model_error
        self.quality_to_error_mapping = quality_to_error_mapping
        self.search_grid = search_grid
        self.signature = signature
        self.comment = comment
        self.travel_time_model_error = travel_time_model_error
        self.phase_identifier_mapping = phase_identifier_mapping
        self.phase_statistics_parameters = phase_statistics_parameters
        self.takeoff_angle_parameters = takeoff_angle_parameters
        self.magnitude_calculation_method = magnitude_calculation_method
        self.magnitude_calculation_component = magnitude_calculation_component
        self.sensor_alias_code = sensor_alias_code
        self.exclude_observations = exclude_observations
        self.phase_time_delay = phase_time_delay
        self.vertical_ray_elevation_correction = \
            vertical_ray_elevation_correction
        self.topography_mask = topography_mask
        self.sensor_distribution_weighting = sensor_distribution_weighting


class Control:
    def __init__(self, message_flag=-1, random_seed=1000):
        """
        Control section
        :param message_flag: (integer, min:-1, default:1) sets the verbosity
        level for messages printed to the terminal ( -1 = completely silent,
        0 = error messages only, 1 = 0 + higher-level warning and progress
        messages, 2 and higher = 1 + lower-level warning and progress
        messages + information messages, ...)
        :param random_seed:(integer) integer seed value for generating random
        number sequences (used by program NLLoc to generate Metropolis samples
        and by program Time2EQ to generate noisy time picks)
        """
        try:
            self.message_flag=int(message_flag)
        except Exception as e:
            raise e

        try:
            self.random_seed=int(random_seed)
        except Exception as e:
            raise e

    def __repr__(self):
        return f'CONTROL {self.message_flag} {self.random_seed}'


class LocSearchGrid:
    def __init__(self, num_sample_draw=1000):
        """

        :param num_sample_draw: specifies the number of scatter samples to
        draw from each saved PDF grid ( i.e. grid with gridType = PROB_DENSITY
        and saveFlag = SAVE ) No samples are drawn if saveFlag < 0.
        :type num_sample_draw: int
        """
        self.num_sample_draw = num_sample_draw

    def __repr__(self):
        return f'GRID {self.num_sample_draw}\n'

    @property
    def type(self):
        return 'LOCSAERCH'


class LocSearchMetropolis:
    def __init__(self, num_samples, num_learn, num_equil, num_begin_save,
                 num_skip, step_init, step_min, prob_min, step_fact=8.):
        """
        Container for the Metropolis Location algorithm parameters

        The Metropolis-Gibbs algorithm performs a directed random walk within
        a spatial, x,y,z volume to obtain a set of samples that follow the
        3D PDF for the earthquake location. The samples give and estimate of
        the optimal hypocenter and an image of the posterior probability
        density function (PDF) for hypocenter location.

        Advantages:

        1. Does not require partial derivatives, thus can be used with
        complicated, 3D velocity structures
        2. Accurate recovery of moderately irregular (non-ellipsoidal)
        PDF's with a single minimum
        3. Only only moderately slower (about 10 times slower) than linearised,
        iterative location techniques, and is much faster
        (about 100 times faster) than the grid-search
        4. Results can be used to obtain confidence contours

        Drawbacks:

        1. Stochastic coverage of search region - may miss important features
        2. Inconsistent recovery of very irregular (non-ellipsoidal)
        PDF's with multiple minima
        3. Requires careful selection of sampling parameters
        4. Attempts to read full 3D travel-time grid files into memory,
        thus may run very slowly with large number of observations and large
        3D travel-time grids

        :param num_samples: total number of accepted samples to obtain (min:0)
        :type num_samples: int
        :param num_learn: number of accepted samples for learning stage of
        search (min:0)
        :type num_learn: int
        :param num_equil: number of accepted samples for equilibration stage
        of search (min:0)
        :type num_equil: int
        :param num_begin_save: number of accepted samples after which to begin
        saving stage of search, denotes end of equilibration stage (min:0)
        :type num_begin_save: int
        :param num_skip: number of accepted samples to skip between saves
        (numSkip = 1 saves every accepted sample, min:1)
        :type num_skip: int
        :param step_init: initial step size in km for the learning stage
        (stepInit < 0.0 gives automatic step size selection. If the search
        takes too long, the initial step size may be too large;
        this may be the case if the search region is very large relative
        to the volume of the high confidence region for the locations.)
        :type step_init: float
        :param step_min: minimum step size allowed during any search stage
        (This parameter should not be critical, set it to a low value. min:0)
        :type step_min: float
        :param prob_min: minimum value of the maximum probability (likelihood)
        that must be found by the end of learning stage, if this value is not
        reached the search is aborted (This parameters allows the filtering of
        locations outside of the search grid and locations with large
        residuals.)
        :type prob_min: float
        :param step_fact: step factor for scaling step size during
        equilibration stage (Try a value of 8.0 to start.) Default=8.
        :type step_fact: float
        """

        self.num_samples = num_samples
        self.num_learn = num_learn
        self.num_equil = num_equil
        self.num_begin_save = num_begin_save
        self.num_skip = num_skip
        self.step_init = step_init
        self.step_min = step_min
        self.prob_min = prob_min
        self.step_fact = step_fact

    def __repr__(self):
        line = f'LOCSEARCH MET {self.num_samples} {self.num_learn} ' \
               f'{self.num_equil} {self.num_begin_save} {self.num_skip} ' \
               f'{self.step_min} {self.step_min} {self.step_fact} ' \
               f'{self.prob_min}\n'

        return line

    @classmethod
    def init_with_default(cls):
        pass

    @property
    def type(self):
        return 'LOCSAERCH'


class LocSearchOctTree:
    def __init__(self, init_num_cell_x, init_num_cell_y, init_num_cell_z,
                 min_node_size, max_node_size, num_scatter,
                 use_station_density=False, stop_on_min_node_size=True):
        """
        Container for the Octree Location algorithm parameters

        Documenation: http://alomax.free.fr/nlloc/octtree/OctTree.html

        Developed in collaboration with Andrew Curtis; Schlumberger Cambridge
        Research, Cambridge CB3 0EL, England; curtis@cambridge.scr.slb.com

        The oct-tree importance sampling algorithm gives accurate, efficient
        and complete mapping of earthquake location PDFs in 3D space (x-y-z).

        Advantages:

        1. Much faster than grid-search (factor 1/100)
        2. More global and complete than Metropolis-simulated annealing
        3. Simple, with very few parameters (initial grid size, number of samples)

        Drawbacks:

        1. Results are weakly dependant on initial grid size - the method may
        not identify narrow, local maxima in the PDF.
        2. Attempts to read full 3D travel-time grid files into memory,
        thus may run very slowly with large number of observations and large
        3D travel-time grids

        :param init_num_cell_x: initial number of octtree cells in the x
        direction
        :type init_num_cell_x: int
        :param init_num_cell_y: initial number of octtree cells in the y
        direction
        :type init_num_cell_y: int
        :param init_num_cell_z: initial number of octtree cells in the z
        direction
        :type init_num_cell_z: int
        :param min_node_size: smallest octtree node side length to process,
        the octree search is terminated after a node with a side smaller
        than this length is generated
        :type min_node_size: float
        :param max_node_size: total number of nodes to process
        :type max_node_size: int
        :param num_scatter: the number of scatter samples to draw from the
        octtree results
        :type num_scatter: int
        :param use_station_density: flag, if True weights oct-tree cell
        probability values used for subdivide decision in proportion to number
        of stations in oct-tree cell; gives higher search priority to cells
        containing stations, stablises convergence to local events when global
        search used with dense cluster of local stations
        (default:False)
        :type use_station_density: bool
        :param stop_on_min_node_size: flag, if True, stop search when first
        min_node_size reached, if False stop subdividing a given cell when
        min_node_size reached (default:True)
        :type stop_on_min_node_size: bool
        """

        self.init_num_cell_x = init_num_cell_x
        self.init_num_cell_y = init_num_cell_y
        self.init_num_cell_z = init_num_cell_z
        self.min_node_size = min_node_size
        self.max_node_size = max_node_size
        self.num_scatter = num_scatter
        self.use_station_density = use_station_density
        self.stop_on_min_node_size = stop_on_min_node_size

    @classmethod
    def init_default(cls):
        init_num_cell_x = 5
        init_num_cell_y = 5
        init_num_cell_z = 5
        min_node_size = 1E-6
        max_node_size = 5000
        num_scatter = 500
        use_station_density = False
        stop_on_min_node_size = True
        return cls(init_num_cell_x, init_num_cell_y, init_num_cell_z,
                   min_node_size, max_node_size, num_scatter,
                   use_station_density, stop_on_min_node_size)

    def __repr__(self):
        line = f'LOCSEARCH OCT {self.init_num_cell_x} ' \
               f'{self.init_num_cell_y} {self.init_num_cell_z} ' \
               f'{self.min_node_size} {self.max_num_nodes} ' \
               f'{self.num_scatter} {self.use_station_density:d} ' \
               f'{self.stop_on_min_node_size:d}\n'

        return line

    @property
    def type(self):
        return 'LOCSEARCH'


class GaussianModelErrors:
    def __init__(self, sigma_time, correlation_length):
        """
        container for Gaussian Error Model
        :param sigma_time: typical error in seconds for travel-time to one
        station due to model errors
        :type sigma_time: float
        :param correlation_length: correlation length that controls covariance
        between stations ( i.e. may be related to a characteristic scale length
        of the medium if variations on this scale are not included in the
        velocity model)
        :type correlation_length: float
        """

        self.sigma_time = sigma_time
        self.correlation_length = correlation_length

    @classmethod
    def init_default(cls):
        sigma_time = 1E-3
        correlation_length = 1E-3

        return cls(sigma_time, correlation_length)

    def __repr__(self):
        return f'LOCGAU {self.sigma_time} {self.correlation_length}\n'



__valid_location_methods__ = ['GAU_ANALYTIC', 'EDT', 'EDT_OT_WT',
                              'EDT_OT_WT_ML']

class LocationMethod:
    def __init__(self, method, max_dist_sta_grid, min_number_phases,
                 max_number_phases, min_number_s_phases, vp_vs_ratio,
                 max_number_3d_grid_memory, min_dist_sta_grid):
        """
        container for location method
        :param method: location method/algorithm ( GAU_ANALYTIC = the inversion
        approach of Tarantola and Valette (1982) with L2-RMS likelihood
        function. EDT = Equal Differential Time likelihood function cast into
        the inversion approach of Tarantola and Valette (1982) EDT_OT_WT =
        Weights EDT-sum probabilities by the variance of origin-time estimates
        over all pairs of readings. This reduces the probability (PDF values)
        at points with inconsistent OT estimates, and leads to more compact
        location PDF's. EDT_OT_WT_ML = version of EDT_OT_WT with EDT
        origin-time weighting applied using a grid-search, maximum-likelihood
        estimate of the origin time. Less efficient than EDT_OT_WT which
        uses simple statistical estimate of the origin time.)
        :param max_dist_sta_grid: maximum distance in km between a station and the
        center of the initial search grid; phases from stations beyond this
        distance will not be used for event location
        :param min_number_phases: minimum number of phases that must be
        accepted before event will be located
        :param max_number_phases: maximum number of accepted phases that will
        be used for event location; only the first maxNumberPhases read from
        the phase/observations file are used for location
        :param min_number_s_phases: minimum number of S phases that must be
        accepted before event will be located
        :param vp_vs_ratio: P velocity to S velocity ratio. If VpVsRatio > 0.0
        then only P phase travel-times grids are read and VpVsRatio is used to
        calculate S phase travel-times. If VpVsRatio < 0.0 then S phase
        travel-times grids are used.
        :param max_number_3d_grid_memory: maximum number of 3D travel-time
        grids to attempt to read into memory for Metropolis-Gibbs search. This
        helps to avoid time-consuming memory swapping that occurs if the total
        size of grids read exceeds the real memory of the computer. 3D grids
        not in memory are read directly from disk. If maxNum3DGridMemory < 0
        then NLLoc attempts to read all grids into memory.
        :param min_dist_sta_grid: minimum distance in km between a station and
        the center of the initial search grid; phases from stations closer than
        this distance will not be used for event location
        """

        validate(method, __valid_location_methods__)
        self.method = method

        self.max_dist_sta_grid = max_dist_sta_grid
        self.min_number_phases = min_number_phases
        self.max_number_phases = max_number_phases
        self.min_number_s_phases = min_number_s_phases
        self.vp_vs_ratio = vp_vs_ratio
        self.max_number_3d_grid_memory = max_number_3d_grid_memory
        self.min_dist_sta_grid = min_dist_sta_grid

    @classmethod
    def init_default(cls):
        method = 'EDT_OT_WT'
        max_dist_sta_grid = 9999
        min_number_phases = 6
        max_number_phases = -1
        min_number_s_phases = -1
        vp_vs_ratio = -1
        max_number_3d_grid_memory = 0
        min_dist_sta_grid = 0

        return cls(method, max_dist_sta_grid, min_number_phases,
                   max_number_phases, min_number_s_phases, vp_vs_ratio,
                   max_number_3d_grid_memory, min_dist_sta_grid)

    def __repr__(self):
        line = f'LOCMETH {self.method} {self.max_dist_sta_grid} ' \
               f'{self.min_number_phases} {self.max_number_phases} ' \
               f'{self.min_number_s_phases} {self.vp_vs_ratio}\n'

        return line



class Observations:
    def __init__(self, picks, p_pick_error=1e-3, s_pick_error=1e-3):
        """

        :param picks: a list of pick object
        :type picks: list of uquake.core.event.pick
        :param p_pick_error: p-wave picking error in second
        :param s_pick_error: s-wave picking error in second
        """

        self.picks = picks
        self.p_pick_error = p_pick_error
        self.s_pick_error = s_pick_error

    @classmethod
    def from_event(cls, event, p_pick_error=1e-3, s_pick_error=1e-3,
                   origin_index=None):

        if type(event) is Catalog:
            event = event[0]
            logger.warning('An object type Catalog was provided. Taking the '
                           'first event of the catalog. This may lead to '
                           'unwanted behaviour')

        if origin_index is None:
            if event.preferred_origin() is None:
                logger.warning('The preferred origin is not defined. The last'
                               'inserted origin will be use. This may lead '
                               'to unwanted behaviour')

                origin = event.origins[-1]
            else:
                origin = event.preferred_origin()
        else:
            origin = event.origins[origin_index]

        picks = [arrival.get_pick() for arrival in origin.arrivals]

        return cls(picks, p_pick_error=p_pick_error,
                   s_pick_error=s_pick_error)

    def __repr__(self):

        lines = ''
        for pick in self.picks:
            if pick.evaluation_status == 'rejected':
                continue

            sensor = pick.sensor
            instrument_identification = pick.waveform_id.channel_code[0:2]
            component = pick.waveform_id.channel_code[-1]
            phase_onset = 'e' if pick.onset in ['emergent', 'questionable'] \
                else 'i'
            phase_descriptor = pick.phase_hint.upper()
            if pick.polarity is None:
                first_motion = '?'
            else:
                first_motion = 'U' if pick.polarity.lower() == 'positive' \
                    else 'D'
            datetime_str = pick.time.strftime('%Y%m%d %H%M %S.%f')

            error_type = 'GAU'
            if pick.phase_hint.upper() == 'P':
                pick_error = f'{self.p_pick_error:0.2e}'
            else:
                pick_error = f'{self.s_pick_error:0.2e}'

            # not implemented
            coda_duration = -1
            amplitude = -1
            period = -1
            phase_weight = 1

            line = f'{sensor:<6s} {instrument_identification:<4s} ' \
                   f'{component:<4s} {phase_onset:1s} ' \
                   f'{phase_descriptor:<6s} {first_motion:1s} ' \
                   f'{datetime_str} {error_type} {pick_error} ' \
                   f'{coda_duration:.2e} {amplitude:.2e} {period:.2e} ' \
                   f'{phase_weight:d}\n'

            lines += line

        return lines

    def write(self, file_name, path='.'):
        with open(Path(path) / file_name) as file_out:
            file_out.write(str(self))


class GridTimeFiles:
    def __init__(self, velocity_file_path, travel_time_file_path, p_wave=True,
                 swap_bytes_on_input=False):
        """
        Specifies the directory path and file root name (no extension), and
        the wave type identifier for the input velocity grid and output
        time grids.
        :param velocity_file_path: full or relative path and file
        root name (no extension) for input velocity grid (generated by
        program Vel2Grid)
        :type velocity_file_path: str
        :param travel_time_file_path: full or relative path and file
        root name (no extension) for output travel-time and take-off angle
        grids
        :type travel_time_file_path: str
        :param p_wave: p-wave if True, s-wave if False
        :type p_wave: bool
        :param swap_bytes_on_input: flag to indicate if hi and low bytes of
        input velocity grid file should be swapped
        :type swap_bytes_on_input: bool
        """

        self.velocity_file_path = velocity_file_path
        self.travel_time_file_path = travel_time_file_path
        if p_wave:
            self.phase = 'P'
        else:
            self.phase = 'S'

        self.swap_bytes_on_input=int(swap_bytes_on_input)

    def __repr__(self):
        return f'GTFILES {self.velocity_file_path} ' \
               f'{self.travel_time_file_path} {self.phase} ' \
               f'{self.swap_bytes_on_input}'


class GridTimeMode:
    def __init__(self, grid_3d=True, calculate_angles=True):
        """
        Specifies several program run modes.
        :param grid_3d: if True 3D grid if False 2D grid
        :type grid_3d: bool
        :param calculate_angles: if True calculate angles and not if False
        :type calculate_angles: bool
        """

        if grid_3d:
            self.grid_mode = 'GRID3D'
        else:
            self.grid_mode = 'GRID2D'

        if calculate_angles:
            self.angle_mode = 'ANGLES_YES'
        else:
            self.angle_mode = 'ANGLES_NO'

    def __repr__(self):
        return f'GTMODE {self.grid_mode} {self.angle_mode}'


class Sensors:
    def __init__(self, sensors):
        """
        specifies a series of source location from an inventory object
        :param sensors: inventory object
        :type sensors: list of uquake.core.inventory.Sensors
        """

        self.sensors = sensors

    @classmethod
    def from_inventory(cls, inventory):
        """
        create from an inventory object
        :param inventory: uquake.core.inventory.Inventory
        :return: a Sensors object
        :rtype: Sensors
        """

        sensors = inventory.sensors
        return cls(sensors)

    def __repr__(self):
        line = ""
        for sensor in self.sensors:

            # test if sensor name is shorter than 6 characters

            line += f'GTSRCE {sensor.code} XYZ ' \
                    f'{sensor.x / 1000:>10.4f} ' \
                    f'{sensor.y / 1000:>10.4f} ' \
                    f'{sensor.z / 1000:>10.4f} ' \
                    f'0.00\n'

        return line


__observation_file_types__ = ['NLLOC_OBS', 'HYPO71', 'HYPOELLIPSE',
                              'NEIC', 'CSEM_ALERT', 'SIMULPS', 'HYPOCENTER',
                              'HYPODD', 'SEISAN', 'NORDIC', 'NCSN_Y2K_5',
                              'NCEDC_UCB', 'ETH_LOC', 'RENASS_WWW',
                              'RENASS_DEP', 'INGV_BOLL', 'INGV_BOLL_LOCAL',
                              'INGV_ARCH']


class ProjectManager:

    inventory_file_name = 'inventory.xml'

    def __init__(self, path, project_name, network_code):
        """
        provide information on file names and locations
        :param path: base path
        :type path: str
        :param project_name: project name or id
        :type project_name: str
        :param network_code: network name or id
        :type network_code: str
        """

        self.root_directory = Path(path) / project_name / network_code
        # create the directory if it does not exist
        self.root_directory.mkdir(parents=True, exist_ok=True)

        self.inventory_location = self.root_directory / 'inventory'
        self.inventory_location.mkdir(parents=True, exist_ok=True)
        self.inventory_file = self.inventory_location / 'inventory.xml'
        if not self.inventory_file.exists():
            logger.warning('the project does not contain an inventory file. '
                           'to add an inventory file use the add_inventory '
                           'method.')

        self.velocity_grid_location = self.root_directory / 'velocities'
        self.velocity_grid_location.mkdir(parents=True, exist_ok=True)

        p_vel_base_name = grid.VelocityGrid3D.get_base_name(self.network, 'P')
        self.p_velocity_file = self.velocity_grid_location / p_vel_base_name
        if not self.p_velocity_file.exists():
            logger.warning('the project does not contain a p-wave velocity '
                           'model. to add a p-wave velocity model to the '
                           'project please use the add_velocity or '
                           'add_velocities methods.')

        s_vel_base_name = grid.VelocityGrid3D.get_base_name(self.network, 'S')
        self.s_velocity_file = self.velocity_grid_location / s_vel_base_name
        if not self.p_velocity_file.exists():
            logger.warning('the project does not contain a s-wave velocity '
                           'model. to add a s-wave velocity model to the '
                           'project please use the add_velocity or '
                           'add_velocities methods.')

        self.travel_time_grid_location = self.root_directory / 'times'
        self.travel_time_grid_location.mkdir(parents=True, exist_ok=True)
        file_list = list(self.travel_time_grid_location / '*.hrd')
        if len(file_list) == 0:
            logger.warning('the project does not contain travel time grids. '
                           'to initialize the travel-time grid use the '
                           'init_travel_time_grid method. Note that '
                           'this require the project to contain both '
                           'an inventory and a velocities files.')

        self.obs_path = self.root_directory / 'observations'
        self.obs_path.mkdir(parents=True, exist_ok=True)

        self.run_id = str(uuid4())
        self.current_run_directory = self.root_directory / 'run' / self.run_id
        self.current_run_directory.mkdir(parents=True, exist_ok=False)

    def init_travel_time_grid(self):
        """
        initialize the travel time grids
        """
        try:
            inventory = read_inventory(self.inventory_file)
        except Exception as e:
            logger.error(e)
            raise IOError('the project does not contain an inventory file. To '
                          'add an inventory file to the project use '
                          'the FileManager.add_inventory method')

        sensors = inventory.sensors

        p_vel_base_name = grid.VelocityGrid3D.get_base_name(self.network, 'P')
        s_vel_base_name = grid.VelocityGrid3D.get_base_name(self.network, 'S')
        try :
            p_velocity = grid.read_grid(self.velocity_grid_location,
                                        p_vel_base_name)
        except Exception as e:
            logger.error(e)
            raise IOError('the project does not contain a P-wave velocity '
                          'model. Use the FileManager.add_velocity or '
                          'FileManager.add_velocities to add a velocity '
                          'model to the project')

        try:
            s_velocity = grid.read_grid(self.velocity_grid_location,
                                        s_vel_base_name)
        except Exception as e:
            logger.error(e)
            raise IOError('the project does not contain a S-wave velocity '
                          'model. Use the FileManager.add_velocity or '
                          'FileManager.add_velocities to add a velocity '
                          'model to the project')

        velocity_models = grid.VelocityGridEnsemble(p_velocity, s_velocity)

        seeds = []
        seed_labels = []
        for sensor in sensors:
            seeds.append(sensor.loc)
            seed_labels(sensor.code)

        tt_gs = velocity_models.to_time_multi_threaded(seeds,seed_labels)
        tt_gs.write(self.travel_time_location)

    def add_inventory(self, inventory):
        """
        adding a inventory object to the project
        :param inventory:
        :return:
        """
        inventory.write(self.inventory_location / self.inventory_file_name)

    def add_velocities(self, velocities):
        """
        add P- and S-wave velocity models to the project
        :param velocities: velocity models
        :type velocities: nlloc.grid.VelocityEnsemble
        :return:
        """

        velocities.write(path=self.velocity_grid_location)

    def add_velocity(self, velocity):
        """
        add P- or S-wave velocity model to the project
        :param velocity: p-wave velocity model
        :type velocity: nlloc.grid.VelocityGrid3D
        :return:
        """

        velocity.write(path=self.velocity_grid_location)

    def add_observations(self, observations):
        """
        adding observations to the project
        :param observations: Observations
        :return:
        """
        self.current_run_directory.mkdir(parents=True, exist_ok=True)
        observations.write('observations.obs', path=self.current_run_directory)

    def clean(self):
        """
        remove the files and directory related to a particular run id and
        create a new run id and related directory.
        :return:
        """
        logger.info(f'removing {self.current_run_directory}')
        for item in list(self.current_run_directory.glob('*')):
            item.unlind()
        self.current_run_directory.rmdir()
        self.run_id = str(uuid4())
        self.current_run_directory = self.root_directory / 'run' / self.run_id

    def add_control_file(self, control, geographic_transformation,
                         search_method, location_method,
                         gaussian_model_errors, signature=None, comment=None):

        """

        :param control:
        :param geographic_transformation:
        :param signature:
        :param comment:
        :return:

        control
        geographic_transformation
        signature
        comment
        locsrce
        locfiles
        locsearch
        locmeth
        locgau
        locgau2 - optional
        locphaseid - optional
        locqualerr - optional
        locgrid

        """

        signature = f'LOCSIG {signature}\n'
        comment = f'LOCCOM {comment}\n'

        ctrl_file = '\n'.join(str(control), str(geographic_transformation),
                              signature, comment, str(search_method),
                              str(location_method), str(gaussian_model_errors))

    def init_default(self):
        Control



class InputFiles:
    def __init__(self, observation_files, travel_time_file_root,
                 output_file_root, observation_file_type='NLLOC_OBS',
                 i_swap_bytes=False, create_missing_folders=True):
        """
        Specifies the directory path and filename for the phase/observation
        files, and the file root names (no extension) for the input time grids
        and the output files.

        the path where the files are to be located is

        :param observation_files: full or relative path and name for
        phase/observations files, mulitple files may be specified with
        standard UNIX "wild-card" characters ( * and ? )
        :type observation_files: str
        :param observation_file_type: (choice: NLLOC_OBS HYPO71 HYPOELLIPSE
        NEIC CSEM_ALERT SIMULPS HYPOCENTER HYPODD SEISAN NORDIC NCSN_Y2K_5
        NCEDC_UCB ETH_LOC RENASS_WWW RENASS_DEP INGV_BOLL
        INGV_BOLL_LOCAL INGV_ARCH) format type for phase/observations files
        (see Phase File Formats) - DEFAULT NLLOC_OBS
        :type observation_file_type: str
        :param travel_time_file_root: full or relative path and file root name
        (no extension) for input time grids (generated by program Grid2Time,
        edu.sc.seis.TauP.TauP_Table_NLL, or other software.
        :type travel_time_file_root: str
        :param output_file_root: full or relative path and file root name
        (no extension) for output files
        :type output_file_root: str
        :param i_swap_bytes: flag to indicate if hi and low bytes of input
        time grid files should be swapped. Allows reading of travel-time grids
        from different computer architecture platforms during TRANS GLOBAL mode
        location. DEFAULT=False
        :type i_swap_bytes: bool
        :param create_missing_folders: if True missing folder will be created
        """

        # validate if the path exist if the path does not exist the path
        # should be created
        observation_files = Path(observation_files)
        if not observation_files.parent.exists():
            if create_missing_folders:
                logger.warning(f'the path <{observation_files.parent}> does '
                               f'not exist. missing folders will be created')
                observation_files.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise IOError(f'path <{observation_files.parent}> does not '
                              f'exist')

        self.observation_files = observation_files

        travel_time_file_root = Path(travel_time_file_root)
        if not travel_time_file_root.parent.exists():
            if create_missing_folders:
                logger.warning(f'the path <{travel_time_file_root.parent}> '
                               f'does not exist. missing folders will '
                               f'be created')

                travel_time_file_root.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise IOError(f'path <{travel_time_file_root.parent}> does '
                              f'not exist')

        self.travel_time_file_root = travel_time_file_root

        output_file_root = Path(output_file_root)

        if not output_file_root.parent.exists():
            if create_missing_folders:
                logger.warning(f'the path <{output_file_root.parent}> '
                               f'does not exist. missing folders will '
                               f'be created')

                output_file_root.parent.mkdir(parents=True, exist_ok=True)
            else:
                raise IOError(f'path <{output_file_root.parent}> does '
                              f'not exist')

        self.output_file_root = output_file_root

        validate(observation_file_type, __observation_file_types__)
        self.observation_file_type = observation_file_type

        self.i_swap_bytes = i_swap_bytes

    def __repr__(self):
        line = f'LOCFILES {self.observation_files} ' \
               f'{self.observation_file_type} {self.travel_time_file_root} ' \
               f'{self.output_file_root} {int(self.i_swap_bytes)}\n'
        return line


class GeographicTransformation:
    type = 'GeographicTransformation'

    def __init__(self, transformation='NONE'):
        validate(transformation, __valid_geographic_transformation__)
        self.transformation = transformation

    def __repr__(self):
        line = f'TRANSFORMATION {self.transformation}'
        return line


class SimpleSDCGeographicTransformation(GeographicTransformation):

    def __init__(self, latitude_origin, longitude_origin,
                 rotation_angle, simple=True):
        """
        The SIMPLE or SDC transformation only corrects longitudinal
        distances as a function of latitude Algorithm:

        >> x = (long - longOrig) * 111.111 * cos(lat_radians);
        >> y = (lat - latOrig) * 111.111;
        >> lat = latOrig + y / 111.111;
        >> long = longOrig + x / (111.111 * cos(lat_radians));

        :param latitude_origin: (float, min:-90.0, max:90.0) latitude in
        decimal degrees of the rectangular coordinates origin
        :param longitude_origin: (float, min:-180.0, max:180.0) longitude in
        decimal degrees of the rectangular coordinates origin
        :param rotation_angle: (float, min:-360.0, max:360.0) rotation angle
        of geographic north in degrees clockwise relative to the rectangular
        coordinates system Y-axis
        :param simple: Transformation is set to SIMPLE if simple is True.
        Transformation is set to SDC if simple is set to False
        """

        if -90 > latitude_origin > 90:
            raise ValueError('latitude_origin must be comprised between '
                             '-90 and 90 degrees')
        if -180 > longitude_origin > 180:
            raise ValueError('longitude_origin must be comprised between '
                             '-180 and 180 degrees')

        if -360 > rotation_angle > 360:
            raise ValueError('the rotation angle must be comprised between '
                             '-360 and 360 degrees')

        self.latitude_origin = latitude_origin
        self.longitude_origin = longitude_origin
        self.rotation_angle = rotation_angle

        if simple:
            transformation = 'SIMPLE'
        else:
            transformation = 'SDC'

        super.__init__(transformation=transformation)

    def __setattr__(self, key, value):
        if key in self.__dict__.keys():
            self.__dict__[key] = value

    def __repr__(self):
        line = f'TRANS {self.transformation} {self.latitude_origin} ' \
               f'{self.longitude_origin} {self.rotation_angle}'

        return line


class LambertGeographicTransformation(GeographicTransformation):
    def __init__(self, reference_ellipsoid, latitude_origin,
                 longitude_origin, first_standard_parallax,
                 second_standard_parallax, rotation_angle):
        """
        Define a Lambert coordinates system for transformation from Lambert
        geographic coordinates to a cartesian/rectangular system.
        :param reference_ellipsoid: (choice: WGS-84 GRS-80 WGS-72
        Australian Krasovsky International Hayford-1909 Clarke-1880
        Clarke-1866 Airy Bessel Hayford-1830 Sphere) reference ellipsoid name
        :param latitude_origin: (float, min:-90.0, max:90.0) latitude in
        decimal degrees of the rectangular coordinates origin
        :param longitude_origin: (float, min:-180.0, max:180.0) longitude in
        decimal degrees of the rectangular coordinates origin
        :param first_standard_parallax: (float, min:-90.0, max:90.0) first
        standard parallels (meridians) in decimal degrees
        :param second_standard_parallax: (float, min:-90.0, max:90.0)
        second standard parallels (meridians) in decimal degrees
        :param rotation_angle: (float, min:-360.0, max:360.0) rotation angle
        of geographic north in degrees clockwise relative to the rectangular
        coordinates system Y-axis
        """

        validate(reference_ellipsoid, __valid_reference_ellipsoid__)

        self.reference_ellipsoid = reference_ellipsoid

        if -90 > latitude_origin > 90:
            raise ValueError('latitude_origin must be comprised between '
                             '-90 and 90 degrees')
        if -180 > longitude_origin > 180:
            raise ValueError('longitude_origin must be comprised between '
                             '-180 and 180 degrees')

        if -360 > rotation_angle > 360:
            raise ValueError('the rotation angle must be comprised between '
                             '-360 and 360 degrees')

        if -90 > first_standard_parallax > 90:
            raise ValueError('first_standard_parallax must be comprised '
                             'between -90 and 90 degrees')

        if -90 > second_standard_parallax > 90:
            raise ValueError('second_standard_parallax must be comprised '
                             'between -90 and 90 degrees')

        self.latitude_origin = latitude_origin
        self.longitude_origin = longitude_origin
        self.rotation_angle = rotation_angle
        self.first_standard_parallax = first_standard_parallax
        self.second_standard_parallax = second_standard_parallax

    def __repr__(self):
        line = f'TRANS LAMBERT {self.reference_ellipsoid} ' \
               f'{self.latitude_origin} {self.longitude_origin} ' \
               f'{self.first_standard_parallax} ' \
               f'{self.second_standard_parallax} {self.rotation_angle}'


class NLLocCtrlFile:
    def __init__(self):
        self.generic_control_statements = {
            'control': '',
            'geographic_transformation': '',
        }

        # not yet implemented
        self.vel2grid = {
            'velocity_output_root': '',
            'wave_type': '',
            'grid_description': '',
            'model_layers': [],
            'model_vertices': []}

        # vel2grid
        self.vel2grid_velocity_output = ''
        self.vel2grid_wave_type = ''
        self.grid_description = ''
        self.velocity_layer = ''
        self.trans_2d_3d = ''
        self.velocity_model_vertex = ''
        self.velocity_model_edge = ''
        self.velocity_model_2d_polygon = ''

        # grid2time
        self.input_output_file_root
        self.program_mode



    # def add_control_section(self, message_flag, random_seed):
    #     """
    #     Sets various general program control parameters.
    #
    #     :param message_flag: (integer, min:-1, default:1) sets the verbosity
    #     level for messages printed to the terminal
    #     ( -1 = completely silent, 0 = error messages only, 1 = 0 +
    #     higher-level warning and progress messages, 2 and higher = 1 +
    #     lower-level warning and progress messages + information messages, ...)
    #
    #     :param random_seed: (integer) integer seed value for generating random
    #     number sequences (used by program NLLoc to generate Metropolis samples
    #     and by program Time2EQ to generate noisy time picks)
    #     """
    #
    #     self.control_section =
    #
    #     if type(message_flag)



__ctrl_file_control__ = 'CONTROL {message_flag} {seed}'

__ctrl_file_geographic_transformation__ = 'TRANS {transform}'

# __ctrl_file_

__ctrl_file_velocity_section__ = """

VGOUT {velocity_output_dir}

"""




control_section = {'verbosity_level': 0}

class CtrlFile:
    def __init__(self, verbosity_level=0, seed=1000, phases=[], ):
        pass


def read_nlloc_hypocenter_file(filename, picks=None,
                               evaluation_mode='automatic',
                               evaluation_status='preliminary'):
    """
    read NLLoc hypocenter file into an events catalog
    :param filename: path to NLLoc hypocenter filename
    :type filename: str
    :return: seismic catalogue
    :rtype: ~uquake.core.event.Catalog
    """
    cat = Catalog()

    with open(filename) as hyp_file:

        all_lines = hyp_file.readlines()
        hyp = [line.split() for line in all_lines if 'HYPOCENTER' in line][0]
        stat = [line.split() for line in all_lines if 'STATISTICS' in line][0]
        geo = [line.split() for line in all_lines if 'GEOGRAPHIC' in line][0]
        qual = [line.split() for line in all_lines if 'QUALITY' in line][0]
        search = [line.split() for line in all_lines if 'SEARCH' in line][0]
        sign = [line.split() for line in all_lines if 'SIGNATURE' in line][0]

        s = int(np.floor(float(geo[7])))
        us = int((float(geo[7]) - s) * 1e6)

        if s < 0:
            s = 0

        if us < 0:
            us = 0

        tme = datetime(int(geo[2]), int(geo[3]), int(geo[4]),
                       int(geo[5]), int(geo[6]), s, us)
        tme = UTCDateTime(tme)

        if 'REJECTED' in all_lines[0]:
            evaluation_status = 'rejected'
            logger.warning('Event located on grid boundary')
        else:
            evaluation_status = evaluation_status

        hyp_x = float(hyp[2]) * 1000
        hyp_y = float(hyp[4]) * 1000
        hyp_z = float(hyp[6]) * 1000

        method = '%s' % ("NLLOC")

        creation_info = obspy.core.event.CreationInfo(
            author='uquake', creation_time=UTCDateTime.now())

        origin = Origin(x=hyp_x, y=hyp_y, z=hyp_z, time=tme,
                        evaluation_mode=evaluation_mode,
                        evaluation_status=evaluation_status,
                        epicenter_fixed=0, method_id=method,
                        creation_info=creation_info)

        xminor = np.cos(float(stat[22]) * np.pi / 180) * np.sin(float(stat[20])
                                                                * np.pi / 180)
        yminor = np.cos(float(stat[22]) * np.pi / 180) * np.cos(float(stat[20])
                                                                * np.pi / 180)
        zminor = np.sin(float(stat[22]) * np.pi / 180)
        xinter = np.cos(float(stat[28]) * np.pi / 180) * np.sin(float(stat[26])
                                                                * np.pi / 180)
        yinter = np.cos(float(stat[28]) * np.pi / 180) * np.cos(float(stat[26])
                                                                * np.pi / 180)
        zinter = np.sin(float(stat[28]) * np.pi / 180)

        minor = np.array([xminor, yminor, zminor])
        inter = np.array([xinter, yinter, zinter])

        major = np.cross(minor, inter)

        major_az = np.arctan2(major[0], major[1])
        major_dip = np.arctan(major[2] / np.linalg.norm(major[0:2]))
        # MTH: obspy will raise error if you try to set float attr to nan below

        if np.isnan(major_az):
            major_az = None

        if np.isnan(major_dip):
            major_dip = None

        # obspy will complain if we use anything other then the exact type
        # it expects. Cannot extend, cannot even import from elsewhere!
        el = obspy.core.event.ConfidenceEllipsoid()
        el.semi_minor_axis_length = float(stat[24]) * 1000
        el.semi_intermediate_axis_length = float(stat[30]) * 1000
        el.semi_major_axis_length = float(stat[32]) * 1000
        el.major_axis_azimuth = major_az
        el.major_axis_plunge = major_dip

        # obsy will complain... see above
        ou = obspy.core.event.OriginUncertainty()
        ou.confidence_ellipsoid = el

        origin.origin_uncertainty = ou

        TravelTime = False
        oq = obspy.core.event.OriginQuality()
        arrivals = []
        stations = []
        phases = []
        oq.associated_phase_count = 0

        for line in all_lines:
            if 'PHASE ' in line:
                TravelTime = True

                continue
            elif 'END_PHASE' in line:
                TravelTime = False

                continue

            if TravelTime:
                tmp = line.split()
                stname = tmp[0]

                phase = tmp[4]
                res = float(tmp[16])
                weight = float(tmp[17])
                sx = float(tmp[18])
                sy = float(tmp[19])
                sz = float(tmp[20])
    # MTH: In order to not get default = -1.0 for ray azimuth + takeoff here, you
    #      need to set ANGLES_YES in the NLLOC Grid2Time control file. Then, when Grid2Time runs, it
    #      will also create the angle.buf files in NLLOC/run/time and when NLLoc runs, it will interpolate
    #      these to get the ray azi + takeoff and put them on the phase line of last.hyp
    # However, the NLLoc generated takeoff angles look to be incorrect (< 90 deg),
    #  likely due to how OT vertical up convention wrt NLLoc.
    # So instead, use the spp generated files *.azimuth.buf and *.takeoff.buf to overwrite these later
    #      15       16       17              18  19       20          21       22     23 24
    #  >   TTpred    Res       Weight    StaLoc(X  Y         Z)        SDist    SAzim  RAz  RDip RQual    Tcorr
    #  >  0.209032  0.002185    1.2627  651.3046 4767.1881    0.9230    0.2578 150.58  -1.0  -1.0  0     0.0000

                azi = float(tmp[22])  # Set to SAzim since that is guaranteed to be set
                # azi = float(tmp[23])
                toa = float(tmp[24])

                dist = np.linalg.norm([sx * 1000 - origin.x,
                                       sy * 1000 - origin.y,
                                       sz * 1000 - origin.z])

                '''
                MTH: Some notes about the NLLOC output last.hyp phase lines:
                    1. SDist - Is just epicentral distance so does not take into account dz (depth)
                               So 3D Euclidean dist as calculated above will be (much) larger
                    2. SAzim - NLLOC manual says this is measured from hypocenter CCW to station
                               But it looks to me like it's actually clockwise!
                    3. RAz - "Ray take−off azimuth at maximum likelihood hypocenter in degrees CCW from North"
                              In a true 3D model (e.g., lateral heterogeneity) this could be different
                              than SAzim.
                              Have to set: LOCANGLES ANGLES_YES 5 to get the angles, otherwise defaults to -1
                              Probably these are also actually measured clockwise from North

                distxy = np.linalg.norm([sx * 1000 - origin.x,
                                         sy * 1000 - origin.y])

                sdist = float(tmp[21])
                sazim = float(tmp[22])
                raz = float(tmp[23])
                rdip = float(tmp[24])

                print("Scan last.hyp: sta:%3s pha:%s dist_calc:%.1f sdist:%.1f sazim:%.1f raz:%.1f rdip:%.1f" % \
                      (stname, phase, distxy, sdist*1e3, sazim, raz, rdip))

                '''

                arrival = Arrival()
                arrival.phase = phase
                arrival.distance = dist
                arrival.time_residual = res
                arrival.time_weight = weight
                arrival.azimuth = azi
                arrival.takeoff_angle = toa
                arrivals.append(arrival)

                for pick in picks:
                    if ((pick.phase_hint == phase) and (
                            pick.waveform_id.station_code == stname)):

                        arrival.pick_id = pick.resource_id.id

                stations.append(stname)
                phases.append(phase)

                oq.associated_phase_count += 1

        stations = np.array(stations)

        points = read_scatter_file(filename.replace('.hyp', '.scat'))

        origin.arrivals = [arr for arr in arrivals]
        origin.scatter = points

        oq.associated_station_count = len(np.unique(stations))

        oq.used_phase_count = oq.associated_phase_count
        oq.used_station_count = oq.associated_station_count
        oq.standard_error = float(qual[8])
        oq.azimuthal_gap = float(qual[12])
        origin.quality = oq

    return origin


def calculate_uncertainty(event, base_directory, base_name, perturbation=5,
                          pick_uncertainty=1e-3):
    """
    :param event: event
    :type event: uquake.core.event.Event
    :param base_directory: base directory
    :param base_name: base name for grids
    :param perturbation:
    :return: uquake.core.event.Event
    """

    if hasattr(event.preferred_origin(), 'scatter'):
        scatter = event.preferred_origin().scatter[:, 1:].copy()
        scatter[:, 0] -= np.mean(scatter[:, 0])
        scatter[:, 1] -= np.mean(scatter[:, 1])
        scatter[:, 2] -= np.mean(scatter[:, 2])
        u, d, v = np.linalg.svd(scatter)
        uncertainty = np.sqrt(d)

        h = np.linalg.norm(v[0, :-1])
        vert = v[0, -1]
        major_axis_plunge = np.arctan2(-vert, h)
        major_axis_azimuth = np.arctan2(v[0, 0], v[0, 1])
        major_axis_rotation = 0

        ce = obspy.core.event.ConfidenceEllipsoid(
             semi_major_axis_length=uncertainty[0],
             semi_intermediate_axis_length=uncertainty[1],
             semi_minor_axis_length=uncertainty[2],
             major_axis_plunge=major_axis_plunge,
             major_axis_azimuth=major_axis_azimuth,
             major_axis_rotation=major_axis_rotation)
        ou = obspy.core.event.OriginUncertainty(confidence_ellipsoid=ce)
        return ou

    narr = len(event.preferred_origin().arrivals)

    # initializing the frechet derivative
    Frechet = np.zeros([narr, 3])

    event_loc = np.array(event.preferred_origin().loc)

    for i, arrival in enumerate(event.preferred_origin().arrivals):
        pick = arrival.pick_id.get_referred_object()
        station_code = pick.waveform_id.station_code
        phase = arrival.phase

        # loading the travel time grid
        filename = '%s/time/%s.%s.%s.time' % (base_directory,
                                              base_name, phase, station_code)

        tt = read_grid(filename, format='NLLOC')
        # spc = tt.spacing

        # build the Frechet derivative

        for dim in range(0, 3):
            loc_p1 = event_loc.copy()
            loc_p2 = event_loc.copy()
            loc_p1[dim] += perturbation
            loc_p2[dim] -= perturbation
            tt_p1 = tt.interpolate(loc_p1, grid_coordinate=False)
            tt_p2 = tt.interpolate(loc_p2, grid_coordinate=False)
            Frechet[i, dim] = (tt_p1 - tt_p2) / (2 * perturbation)

    hessian = np.linalg.inv(np.dot(Frechet.T, Frechet))
    tmp = hessian * pick_uncertainty ** 2
    w, v = np.linalg.eig(tmp)
    i = np.argsort(w)[-1::-1]
    # for the angle calculation see
    # https://en.wikipedia.org/wiki/Euler_angles#Tait-Bryan_angles
    X = v[:, i[0]]  # major
    Y = v[:, i[1]]  # intermediate
    # Z = v[:, i[2]]  # minor

    X_H = np.sqrt(X[0] ** 2 + X[1] ** 2)
    major_axis_plunge = np.arctan2(X[2], X_H)
    major_axis_azimuth = np.arctan2(X[1], X[0])
    major_axis_rotation = 0

    # major_axis_plunge = np.arcsin(X[2] / np.sqrt(1 - X[2] ** 2))
    # major_axis_azimuth = np.arcsin(X[1] / np.sqrt(1 - X[2] ** 2))
    # major_axis_rotation = np.arcsin(-X[2])
    ce = obspy.core.event.ConfidenceEllipsoid(
        semi_major_axis_length=w[i[0]],
        semi_intermediate_axis_length=w[i[1]],
        semi_minor_axis_length=w[i[2]],
        major_axis_plunge=major_axis_plunge,
        major_axis_azimuth=major_axis_azimuth,
        major_axis_rotation=major_axis_rotation)
    ou = obspy.core.event.OriginUncertainty(confidence_ellipsoid=ce)

    return ou


def read_scatter_file(filename):
    """
    :param filename: name of the scatter file to read
    :return: a numpy array of the points in the scatter file
    """

    f = open(filename, 'rb')

    nsamples = unpack('i', f.read(4))[0]
    unpack('f', f.read(4))
    unpack('f', f.read(4))
    unpack('f', f.read(4))

    points = []

    for k in range(0, nsamples):
        x = unpack('f', f.read(4))[0] * 1000
        y = unpack('f', f.read(4))[0] * 1000
        z = unpack('f', f.read(4))[0] * 1000
        pdf = unpack('f', f.read(4))[0]

        points.append([x, y, z, pdf])

    return np.array(points)


def is_supported_nlloc_grid_type(grid_type):
    """
    verify that the grid_type is a valid NLLoc grid type
    :param grid_type: grid_type
    :type grid_type: str
    :rtype: bool
    """
    grid_type = grid_type.upper()

    if grid_type in supported_nlloc_grid_type:
        return True

    return False


def _read_nll_header_file(file_name):
    """
    read NLLoc header file
    :param file_name: path to the header file
    :type file_name: str
    :rtype: ~uquake.core.AttribDict
    """
    dict_out = AttribDict()
    with open(file_name, 'r') as fin:
        line = fin.readline()
        line = line.split()
        dict_out.shape = tuple([int(line[0]), int(line[1]), int(line[2])])
        dict_out.origin = np.array([float(line[3]), float(line[4]),
                                    float(line[5])])
        dict_out.origin *= 1000
        dict_out.spacing = float(line[6]) * 1000
        dict_out.grid_type = line[9]

        line = fin.readline()

        if dict_out.grid_type in ['ANGLE', 'TIME']:
            line = line.split()
            dict_out.label = line[0]
            dict_out.seed = (float(line[1]) * 1000,
                             float(line[2]) * 1000,
                             float(line[3]) * 1000)

        else:
            dict_out.label = None
            dict_out.seed = None

    return dict_out


def read_NLL_grid(base_name):
    """
    read NLL grids into a GridData object
    :param base_name: path to the file excluding the extension. The .hdr and
    .buf extensions are added automatically
    :type base_name: str
    :rtype: ~uquake.core.data.grid.GridDataa

    .. NOTE:
        The function detects the presence of either the .buf or .hdr extensions
    """

    from uquake.core import GridData
    # Testing the presence of the .buf or .hdr extension at the end of
    # base_name

    if ('.buf' == base_name[-4:]) or ('.hdr' == base_name[-4:]):
        # removing the extension
        base_name = base_name[:-4]

    # Reading header file
    try:
        head = _read_nll_header_file(base_name + '.hdr')
    except ValueError:
        logger.error('error reading %s' % base_name + '.hdr')

    # Read binary buffer
    gdata = np.fromfile(base_name + '.buf', dtype=np.float32)
    gdata = gdata.reshape(head.shape)

    if head.grid_type == 'SLOW_LEN':
        gdata = head.spacing / gdata
        head.grid_type = 'VELOCITY'

    return GridData(gdata, spacing=head.spacing, origin=head.origin,
                    seed=head.seed, seed_label=head.label,
                    grid_type=head.grid_type)


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


def _write_grid_header(base_name, shape, origin, spacing, grid_type,
                       station=None, seed=None):
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


def write_nll_grid(base_name, data, origin, spacing, grid_type, seed=None,
                   label=None, velocity_to_slow_len=True):
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

    if not is_supported_nlloc_grid_type(grid_type):
        logger.warning('Grid type is not a valid NLLoc type')

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


# def prepare_nll(ctl_filename='input.xml', nll_base='NLL'):
#     """
#     :param ctl_filename: path to the XML file containing control parameters
#     :param nll_base: directory in which NLL project will be built
#     """
#     params = ctl.parseControlFile(ctl_filename)
#     keys = ['velgrids', 'sensors']
#     for job_index, job in enumerate(ctl.buildJob(keys, params)):
#
#         params = ctl.getCurrentJobParams(params, keys, job)
#         nll_opts = init_from_xml_params(params, base_folder=nll_base)
#         nll_opts.prepare(create_time_grids=True, tar_files=False)


def init_nlloc_from_params(params):
    """

    """
    project_code = params.project_code

    nll = NLL(project_code, base_folder=params.nll.NLL_BASE)
    nll.gridpar = params.velgrids
    nll.sensors = params.sensors
    nll.params = params.nll

    nll.hdrfile.gridpar = nll.gridpar.grids.vp
    nll.init_control_file()

    return nll


class NLL(object):

    def __init__(self, project_code, base_folder='NLL', gridpar=None,
                 sensors=None, params=None):
        """
        :param project_code: the name of project, to be used for generating
        file names
        :type project_code: str
        :param event: and event containing picks and an origin with arrivals
        referring to the picks
        :type event: ~uquake.core.event.Event
        :param base_folder: the name of the NLL folder
        :type base_folder: str
        """
        self.project_code = project_code
        self.base_folder = base_folder

        self.ctrlfile = NLLControl()
        self.hdrfile = NLLHeader()

        self.gridpar = gridpar
        self.sensors = sensors
        self.params = params

        self.hdrfile.gridpar = self.gridpar.grids.vp
        self.init_control_file()

    @property
    def base_name(self):
        return '%s' % self.project_code

    def _make_base_folder(self):
        try:
            if not os.path.exists(self.base_folder):
                os.mkdir(self.base_folder)

            if not os.path.exists(os.path.join(self.base_folder, 'run')):
                os.mkdir(os.path.join(self.base_folder, 'run'))

            if not os.path.exists(os.path.join(self.base_folder, 'model')):
                os.mkdir(os.path.join(self.base_folder, 'model'))

            if not os.path.exists(os.path.join(self.base_folder, 'time')):
                os.mkdir(os.path.join(self.base_folder, 'time'))

            return True
        except:
            return False

    def _clean_outputs(self):
        try:
            for f in glob(os.path.join(self.base_folder, 'loc',
                                       self.base_name)):
                os.remove(f)
        except:
            pass

    def _prepare_project_folder(self):

        self.worker_folder = tempfile.mkdtemp(dir=self.base_folder).split(
            '/')[-1]

        os.mkdir(os.path.join(self.base_folder, self.worker_folder, 'loc'))
        os.mkdir(os.path.join(self.base_folder, self.worker_folder, 'obs'))
        logger.debug('%s.%s: cwd=%s' % (__name__, '_prepare_project_folder',
                                        os.getcwd()))

    def _finishNLL(self):
        '''
        file = "%s/run/%s_%s.in" % (self.base_folder, self.base_name,
        self.worker_folder)
        print("_finishNLL: Don't remove tmp=%s/%s" % (self.base_folder,
        self.worker_folder))
        return
        '''

        os.remove('%s/run/%s_%s.in' % (self.base_folder, self.base_name,
                                       self.worker_folder))
        self._clean_outputs()
        tmp = '%s/%s' % (self.base_folder, self.worker_folder)
        shutil.rmtree(tmp)

    def init_header_file(self):
        """
        """
        pass

    def init_control_file(self):
        """
        """
        self.ctrlfile.vggrid = "VGGRID %s" % (str(self.hdrfile))

        if self.gridpar.homogeneous:
            laymod = "LAYER    %f  %f 0.00    %f  0.00  2.7 0.0" % (
                self.gridpar.grids.vp.origin[2] / 1000,
                self.gridpar.vp / 1000,
                self.gridpar.vs / 1000)

            modelname = self.project_code
        else:
            laymod = "LAYER"
            modelname = self.project_code

        modelname = '%s' % modelname

        self.ctrlfile.laymod = laymod
        self.ctrlfile.modelname = modelname
        self.ctrlfile.basefolder = self.base_folder

        # hdr = "%d %d %d  %.2f %.2f %.2f  %.4f %.4f %.4f  SLOW_LEN" % (
        self.ctrlfile.locgrid = "LOCGRID  %d %d %d  %.2f %.2f %.2f  %.4f " \
                                "%.4f %.4f  MISFIT  SAVE" % (
                                    (self.gridpar.grids.vp.shape[0] - 1) * 10 + 1,
                                    (self.gridpar.grids.vp.shape[1] - 1) * 10 + 1,
                                    (self.gridpar.grids.vp.shape[2] - 1) * 10 + 1,
                                    self.gridpar.grids.vp.origin[0] / 1000,
                                    self.gridpar.grids.vp.origin[1] / 1000,
                                    self.gridpar.grids.vp.origin[2] / 1000,
                                    self.gridpar.grids.vp.spacing / 10000,
                                    self.gridpar.grids.vp.spacing / 10000,
                                    self.gridpar.grids.vp.spacing / 10000)

        self.ctrlfile.locsig = self.params.locsig
        self.ctrlfile.loccom = self.params.loccom
        self.ctrlfile.locsearch = self.params.locsearch
        self.ctrlfile.locmeth = self.params.locmeth

        self.ctrlfile.phase = 'P'
        self.ctrlfile.vgtype = 'P'

        self.ctrlfile.basefolder = self.base_folder
        self.ctrlfile.projectcode = self.project_code

        try:
            self.ctrlfile.add_stations(self.sensors.name, self.sensors.pos)
        except:
            logger.error('Sensor file does not exist')

    def _write_velocity_grids(self):
        if not self.gridpar.homogeneous:
            if self.gridpar.vp:
                p_file = '%s/model/%s.P.mod' % (self.base_folder,
                                                self.base_name)
                self.gridpar.grids.vp.write(p_file, format='NLLOC')
                self.gridpar.filep = self.gridpar.vs.split('/')[-1]
            else:
                self.gridpar.filep = None

            if self.gridpar.vs:
                s_file = '%s/model/%s.S.mod' % (self.base_folder,
                                                self.base_name)
                self.gridpar.grids.vs.write(s_file, format='NLLOC')

                self.gridpar.files = self.gridpar.vs.split('/')[-1]
            else:
                self.gridpar.files = None

        if self.gridpar.homogeneous:
            self.ctrlfile.vgout = '%s/model/%s' % (self.base_folder,
                                                   self.base_name)
            self.ctrlfile.vgout = '%s/model/%s' % (self.base_folder,
                                                   self.base_name)

        else:
            self.ctrlfile.vgout = '%s/model/%s.P.mod.buf' % (self.base_folder,
                                                             self.base_name)
            self.ctrlfile.vgout = '%s/model/%s.S.mod.hdr' % (self.base_folder,
                                                             self.base_name)

    def prepare(self, create_time_grids=True, create_angle_grids=True,
                create_distance_grids=False, tar_files=False):
        """
        Creates the NLL folder and prepare the NLL configuration files based
        on the given configuration

        :param create_time_grids: if True, runs Vel2Grid and Grid2Time
        :type create_time_grids: bool
        :param tar_files: creates a tar of the NLL library
        :type tar_files: bool
        """

        logger.debug(os.getcwd())
        self._make_base_folder()
        logger.debug(os.getcwd())

        self.hdrfile.write('%s/run/%s.hdr' % (self.base_folder,
                                              self.base_name))
        self._write_velocity_grids()
        self.ctrlfile.write('%s/run/%s.in' % (self.base_folder,
                                              self.base_name))

        if create_time_grids:
            self._create_time_grids()

        if create_angle_grids:
            self._create_angle_grids()

        if create_distance_grids:
            self._create_distance_grids()

        if tar_files:
            self.tar_files()

    def _create_time_grids(self):
        self.ctrlfile.phase = 'P'
        self.ctrlfile.vgtype = 'P'
        self.ctrlfile.write('%s/run/%s.in' % (self.base_folder,
                                              self.base_name))

        if self.gridpar.vp:
            if self.gridpar.homogeneous:
                logger.debug('Creating P velocity grid')
                cmd = 'Vel2Grid %s/run/%s.in' % (self.base_folder,
                                                 self.base_name)
                os.system(cmd)

            logger.debug('Calculating P time grids')
            cmd = 'Grid2Time %s/run/%s.in' % (self.base_folder, self.base_name)
            os.system(cmd)

        if self.gridpar.vs:
            self.ctrlfile.phase = 'S'
            self.ctrlfile.vgtype = 'S'
            self.ctrlfile.write('%s/run/%s.in' % (self.base_folder,
                                                  self.base_name))

            if self.gridpar.homogeneous:
                logger.debug('Creating S velocity grid')
                cmd = 'Vel2Grid %s/run/%s.in' % (self.base_folder,
                                                 self.base_name)
                os.system(cmd)

            logger.debug('Calculating S time grids')
            cmd = 'Grid2Time %s/run/%s.in' % (self.base_folder, self.base_name)
            os.system(cmd)

    def _create_angle_grids(self):
        """
        calculate and write angle grids from travel time grids
        """

        time_files = glob('%s/time/*time*.hdr' % self.base_folder)

        for time_file in time_files:
            self._save_angle_grid(time_file)
        # map(self._save_angle_grid, time_files)

    def _save_angle_grid(self, time_file):
        """
        calculate and save take off angle grid
        """
        from uquake.core.simul.eik import angles
        # reading the travel time grid
        ifile = time_file
        ttg = read_grid(ifile, format='NLLOC')
        az, toa = angles(ttg)
        tmp = ifile.split('/')
        tmp[-1] = tmp[-1].replace('time', 'take_off')
        ofname = '/'.join(tmp)
        toa.write(ofname, format='NLLOC')
        az.write(ofname.replace('take_off', 'azimuth'), format='NLLOC')

    def _create_distance_grids(self):
        """
        create distance grids using the ray tracer. Will take long time...
        Returns:

        """
        from uquake.core.simul.eik import ray_tracer
        time_files = glob('%s/time/*time*.hdr' % self.base_folder)

        ttg = read_grid(time_files[0], format='NLLOC')
        x = np.arange(0, ttg.shape[0])
        y = np.arange(0, ttg.shape[1])
        z = np.arange(0, ttg.shape[2])

        X, Y, Z = np.meshgrid(x, y, z)
        X = X.reshape(np.product(ttg.shape))
        Y = Y.reshape(np.product(ttg.shape))
        Z = Z.reshape(np.product(ttg.shape))

        out_array = np.zeros_like(ttg.data)

        for time_file in time_files:
            ttg = read_grid(time_file, format='NLLOC')

            for coord in zip(X, Y, Z):
                st = time()
                ray = ray_tracer(ttg, coord, grid_coordinates=True,
                                 max_iter=100)
                et = time()
                #print(et - st)
                out_array[coord[0], coord[1], coord[2]] = ray.length

            tmp = time_file.split('/')
            tmp[-1] = tmp[-1].replace('time', 'distance')
            ofname = '/'.join(tmp)

            ttg.type = 'DISTANCE'
            ttg.write(ofname, format='NLLOC')

            return

    def tar_files(self):
        # Create tar.gz from the NLL folder
        script = """
        tar -czvf NLL.tar.gz %s/*
        """ % (self.base_folder)

        with open('tmp.sh', 'w') as shfile:
            shfile.write(script)

        logger.debug('Preparing NLL tar file...')
        os.system('sh tmp.sh')
        os.remove('tmp.sh')

    def run_event(self, event, silent=True):
        fname = 'run_event'

        evt = event

        self._prepare_project_folder()

        # TODO
        # MTH: If input event has no preferred_origin(), gen_observations
        # will (incorrectly) create one!
        event2 = self.gen_observations_from_event(evt)

        new_in = '%s/run/%s_%s.in' % (self.base_folder, self.base_name,
                                      self.worker_folder)
        # print("new_in=%s" % new_in)

        self.ctrlfile.workerfolder = self.worker_folder
        self.ctrlfile.write(new_in)

        os.system('NLLoc %s' % new_in)

        filename = "%s/%s/loc/last.hyp" % (self.base_folder,
                                           self.worker_folder)
        logger.debug('%s.%s: scan hypo from filename = %s' % (__name__,
                                                              fname, filename))

        if not glob(filename):
            logger.error("%s.%s: location failed" % (__name__, fname))
            return Catalog(events=[evt])

        if event.origins:
            if event.preferred_origin():
                logger.debug('%s.%s: event.pref_origin exists --> set eval '
                             'mode' % (__name__, fname))
                evaluation_mode = event.preferred_origin().evaluation_mode
                evaluation_status = event.preferred_origin().evaluation_status
            else:
                logger.debug(
                    '%s.%s: event.pref_origin does NOT exist --> set eval '
                    'mode on origins[0]' % (__name__, fname))
                evaluation_mode = event.origins[0].evaluation_mode
                evaluation_status = event.origins[0].evaluation_status

        cat_out = self.read_hyp_loc(filename, event=event,
                                    evaluation_mode=evaluation_mode,
                                    evaluation_status=evaluation_status)

        self._finishNLL()
        return cat_out

    def gen_observations_from_event(self, event):
        """
        Create NLLoc compatible observation file from an uquake event
        catalog file.
        input:

        :param event: event containing a preferred origin with arrivals
        referring to picks
        :type event: ~uquake.core.event.Event
        """

        fname = 'gen_observations_from_event'

        with open('%s/%s/obs/%s.obs' % (self.base_folder, self.worker_folder,
                                        self.base_name), 'w') as out_file:
            po = event.preferred_origin()
            logger.debug('%s.%s: pref origin=[%s]' % (__name__, fname, po))

            if not po:
                logger.error('preferred origin is not set')

            for arr in po.arrivals:

                pk = arr.pick_id.get_referred_object()
                # logger.debug(pk)
                if pk.evaluation_status == 'rejected':
                    continue

                date_str = pk.time.strftime('%Y%m%d %H%M %S.%f')

                if pk.phase_hint == 'P':
                    pick_error = '1.00e-03'
                else:
                    pick_error = '1.00e-03'

                polarity = 'U' if pk.polarity == 'positive' else 'D'

                out_file.write(
                    '%s ?    ?    ?    %s %s %s GAU'
                    ' %s -1.00e+00 -1.00e+00 -1.00e+00\n' % (
                        pk.waveform_id.station_code.ljust(6),
                        pk.phase_hint.ljust(6), polarity, date_str,
                        pick_error))
        return event

    def read_hyp_loc(self, hypfile, event, evaluation_mode='automatic',
                     evaluation_status='preliminary', use_ray_tracer=True):
        """
        read the hypocenter file generate by the location run
        :param hypfile: path to hypocenter file generated by the NLLoc location
        run
        :type hypfile: str
        :param event: an event object with picks
        :type event: uquake.core.Event.event
        :param evaluation_mode: evaluation mode
        :type evaluation_mode: str
        :param evaluation_status: evaluation status
        :type evaluation_status: str
        :param use_ray_tracer: if true use ray tracer to measure
        event-station distance (default: True)
        :type use_ray_tracer: bool
        :rtype: ~uquake.core.event.Catalog
        """
        from uquake.core.simul.eik import ray_tracer
        from time import time

        origin = read_nlloc_hypocenter_file(hypfile, event.picks,
                                            evaluation_mode=evaluation_mode,
                                            evaluation_status=evaluation_status)

        logger.info('ray tracing')
        st = time()
        if use_ray_tracer:
            for arrival in origin.arrivals:
                try:
                    sensor_id = arrival.get_pick().waveform_id.station_code
                    phase = arrival.phase

                    fname = '%s.%s.%s.time' % (self.base_name, phase,
                                               sensor_id)

                    fpath = os.path.join(self.base_folder, 'time', fname)

                    ttg = read_grid(fpath, format='NLLOC')
                    ray = ray_tracer(ttg, origin.loc, grid_coordinates=False)

                    '''
                    dist = arrival.distance
                    pk = arrival.pick_id.get_referred_object()
                    sta = pk.waveform_id.station_code
                    '''
                    arrival.distance = ray.length
                except Exception as exc:
                    logger.warning(
                        f'Failed to calculate ray for sensor {sensor_id}'
                        f' phase {phase}: {exc}', exc_info=True)
                    arrival.distance = None

        et = time()
        logger.info('completed ray tracing in %0.3f' % (et - st))

        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id

        return Catalog(events=[event])

    def take_off_angle(self, station):
        fname = '%s/time/%s.P.%s.take_off' % (self.base_folder, self.base_name,
                                              station)
        return read_grid(fname, format='NLLOC')


class NLLHeader(AttribDict):

    attributes = ['gridpar']

    def __str__(self):
        gridpar = self.gridpar
        hdr = "%d %d %d  %.4f %.4f %.4f  %.4f %.4f %.4f  SLOW_LEN" % (
            gridpar.shape[0],
            gridpar.shape[1],
            gridpar.shape[2],
            gridpar.origin[0] / 1000.,
            gridpar.origin[1] / 1000.,
            gridpar.origin[2] / 1000.,
            gridpar.spacing / 1000.,
            gridpar.spacing / 1000.,
            gridpar.spacing / 1000.)
        # hdr = self.__hdr_tmpl.replace(token,hdr)
        return hdr

    def __init__(self, *args, **kwargs):
        super(NLLHeader, self).__init__(*args, **kwargs)
        for attr in self.attributes:
            self[attr] = ''

    def read(self, fname):
        with open(fname, 'r') as fin:
            line = fin.readline()
            line = line.split()
            self.gridpar = AttribDict()
            self.gridpar.grids = AttribDict()
            self.gridpar.grids.v = AttribDict()
            self.gridpar.shape = tuple([int(line[0]), int(line[1]),
                                        int(line[2])])
            self.gridpar.origin = np.array([float(line[3]), float(line[4]),
                                            float(line[5])])
            self.gridpar.origin *= 1000
            self.gridpar.spacing = float(line[6]) * 1000

    def write(self, fname):
        with open(fname, 'w') as fout:
            token = '<HDR>'
            hdr = self.__hdr_tmpl.replace(token, str(self))
            fout.write(hdr)

    __hdr_tmpl = \
        """<HDR>
TRANSFORM  NONE
"""


supported_nlloc_grid_type = ['VELOCITY', 'VELOCITY_METERS', 'SLOWNESS',
                             'SLOW_LEN', 'TIME', 'PROB_DENSITY', 'MISFIT',
                             'ANGLE', ]


valid_nlloc_grid_type = ['VELOCITY', 'VELOCITY_METERS', 'SLOWNESS', 'VEL2',
                         'SLOW2', 'SLOW2_METERS', 'SLOW_LEN', 'TIME', 'TIME2D',
                         'PROB_DENSITY', 'MISFIT', 'ANGLE', 'ANGLE2D']


class NLLControl(AttribDict):
    """
    NLLoc control file builder
    """

    tokens = ['workerfolder', 'projectcode', 'basefolder', 'modelname',
              'vgout', 'vgtype', 'vggrid', 'laymod',
              'loccom', 'locsig', 'locsearch',
              'locgrid', 'locmeth', 'modelname',
              'phase', 'gtsrce']

    def __str__(self):
        ctrl = self.__ctrl_tmpl
        for attr in self.tokens:
            token = '<%s>' % attr.upper()
            ctrl = ctrl.replace(token, self.__dict__[attr])
        return ctrl

    def __init__(self, *args, **kwargs):
        super(NLLControl, self).__init__(*args, **kwargs)
        for attr in self.tokens:
            self[attr] = ''

    def add_stations(self, sname, sloc):

        for n, l in zip(sname, sloc):
            l2 = l / 1000
            if len(n) > 6:
                logger.critical('NLL cannot handle station names longer than'
                                ' 6 characters, Sensor %s currently has %d'
                                ' characters' % (n, len(n)))
                logger.warning('Sensor %s will not be processed' % n)
                continue
            # noinspection PyStringFormat
            self.gtsrce += 'GTSRCE %s XYZ %f %f %f 0.00\n' % ((n,) + tuple(l2))

    def write(self, fname):
        with open(fname, 'w') as fout:
            fout.write(str(self))

    __ctrl_tmpl = \
"""
CONTROL 0 54321
TRANS NONE
VGOUT  <VGOUT> #<BASEFOLDER>/model/layer

VGTYPE P
VGTYPE S

<VGGRID>

<LAYMOD>

GTFILES  <BASEFOLDER>/model/<MODELNAME>  <BASEFOLDER>/time/<MODELNAME> <PHASE>

GTMODE GRID3D ANGLES_NO
# MTH Uncomment these if you want Grid2Time to calculate angles.buf (takeoff + azimuth)
#     and for the resulting angles to appear on the last.hyp phase lines
#GTMODE GRID3D ANGLES_YES
#LOCANGLES ANGLES_YES 5

<GTSRCE>

GT_PLFD  1.0e-3  0

LOCSIG Microquake package

LOCCOM created automatically by the uquake package

LOCFILES <BASEFOLDER>/<WORKERFOLDER>/obs/<MODELNAME>.obs NLLOC_OBS <BASEFOLDER>/time/<MODELNAME>  <BASEFOLDER>/<WORKERFOLDER>/loc/<MODELNAME>

#LOCHYPOUT SAVE_NLLOC_ALL SAVE_HYPOINV_SUM SAVE_NLLOC_OCTREE
LOCHYPOUT SAVE_NLLOC_ALL

LOCSEARCH <LOCSEARCH>

<LOCGRID>

LOCMETH <LOCMETH>

LOCGAU 0.001 0

LOCGAU2 0.001 0.001 0.001

LOCPHASEID  P   P p
LOCPHASEID  S   S s

LOCQUAL2ERR 0.0001 0.0001 0.0001 0.0001 0.0001

LOCPHSTAT 9999.0 -1 9999.0 1.0 1.0 9999.9 -9999.9 9999.9
"""
