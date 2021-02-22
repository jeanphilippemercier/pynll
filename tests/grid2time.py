from nlloc import nlloc
from importlib import reload
from uquake.core.inventory import read_inventory
import os
from pathlib import Path
# from nlloc.grid import (NLLocGrid, LayeredVelocityModel, ModelLayer,
#                         VelocityGrid3D)

from nlloc import grid
from time import time
from loguru import logger

import numpy as np

import timeit
reload(nlloc)
reload(grid)

test_artifacts = os.environ['UQUAKE_TEST_ARTIFACTS']
inventory_file = test_artifacts + '/inventory.xml'

origin = [650200, 4766170, -500]
dimensions = [100, 101, 68]
spacing = [25, 25, 25]

project_code = 'TEST'
network_code = project_code

# p_grid = NLLocGrid(base_name, dimensions, origin, spacing, 'P',
#                    seed=None, seed_label=None,
#                    grid_type='VELOCITY_METERS', grid_units='METER',
#                    float_type="FLOAT", model_id=None)
#
# s_grid = NLLocGrid(base_name, dimensions, origin, spacing, 'S',
#                    seed=None, seed_label=None,
#                    grid_type='VELOCITY_METERS', grid_units='METER',
#                    float_type="FLOAT", model_id=None)

z = [1168, 459, -300, -500]
vp_z = [4533, 5337, 5836, 5836]
vs_z = [2306, 2885, 3524, 3524]

layered_model_p = grid.LayeredVelocityModel(phase='P')
layered_model_s = grid.LayeredVelocityModel(phase='S')

for (z_, vp, vs) in zip(z, vp_z, vs_z):
    layer_p = grid.ModelLayer(z_, vp)
    layered_model_p.add_layer(layer_p)
    layer_s = grid.ModelLayer(z_, vs)
    layered_model_s.add_layer(layer_s)

base_directory = Path(test_artifacts) / 'vel2grid'

vel_3d_p = layered_model_p.gen_3d_grid(network_code, dimensions, origin,
                                       spacing)
vel_3d_s = layered_model_s.gen_3d_grid(network_code, dimensions, origin,
                                       spacing)

vel_3d_p.write(project_code)

slow_lens_3d_p = vel_3d_p.to_slow_lens()
slow_lens_3d_s = vel_3d_s.to_slow_lens()

slow_lens_3d_p.write(project_code, base_directory / project_code / 'models')
slow_lens_3d_s.write(project_code, base_directory / project_code / 'models')

inventory = read_inventory(inventory_file)

seed = inventory.sensors[10].loc
seed_label = inventory.sensors[10].code

nb = 10
t0 = time()
# for i in range(0, nb):
#     tt = vel_3d_s.to_time(seed, seed_label)
t1 = time()
print((t1 - t0) / nb)

seeds = []
seed_labels = []
for sensor in inventory.sensors:
    seeds.append(sensor.loc)
    seed_labels.append(sensor.code)

t0 = time()
travel_time_grids_s = vel_3d_s.to_time_multi_threaded(seeds, seed_labels)
travel_time_grids_p = vel_3d_s.to_time_multi_threaded(seeds, seed_labels)
t1 = time()
logger.info(f'done calculating the travel time grids in {t1-t0:0.2f} second')

travel_time_grids = travel_time_grids_p + travel_time_grids_s

t0 = time()
rays = travel_time_grids.ray_tracer(np.array(origin) + 200)
t1 = time()
logger.info(f'done calculating the rays in {t1-t0:0.2f} second')

# ray = results[0].ray_tracer([10, 10, 10], grid_coordinate=True)


# timeit.timeit(vel_3d_p.eikonal(seed, seed_label), number=10)

# g2t = nlloc.Grid2Time(inventory, base_directory=base_directory,
#                       base_name=base_name, verbosity=0)
