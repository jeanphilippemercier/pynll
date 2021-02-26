from nlloc import nlloc, grid
from importlib import reload
from loguru import logger

try:
    from .fixtures.data import (inventory, catalog, get_inventory)
    from .fixtures.grids import (travel_time_grids, velocity_grids,
                                 get_travel_time_grids,
                                 get_velocity_grids)
except Exception as e:
    logger.error(e)
    from fixtures.data import (inventory, catalog, get_inventory)
    from fixtures.grids import (travel_time_grids, velocity_grids,
                                 get_travel_time_grids,
                                 get_velocity_grids)

from pathlib import Path
import pytest
import numpy as np

from uquake.core.grid import read_grid

reload(nlloc)
project_dir = 'NLL'


root_path = Path('TEST/NLL')
project = 'TEST'
observation_dir = 'obs'
travel_time_dir = 'time'
output_file_dir = 'hyp'
model_file_dir = 'model'

velocity_grid_file = root_path / model_file_dir
observation_files = root_path / observation_dir / 'test'
travel_time_root = root_path / travel_time_dir
output_file_root = root_path / output_file_dir
control = nlloc.Control()

geographic_transformation = nlloc.GeographicTransformation()
sensors = nlloc.Sensors.from_inventory(get_inventory())
files = nlloc.InputFiles(observation_files, travel_time_root,
                         output_file_root)

# tt_grids = get_travel_time_grids()

# p_velocity = grid.read_grid('TEST.P.mod', path=velocity_grid_file)


def write_travel_time_grids():
    tt_grids = get_travel_time_grids()
    tt_grids.write(travel_time_root)


def write_velocity_grids():
    velocity_grids = get_velocity_grids()
    velocity_grids.write(path=velocity_grid_file)


def test_create_observations(catalog):
    obs = nlloc.Observations.from_event(catalog)

    with open(observation_files, 'w') as obs_file:
        obs_file.write(str(obs))

    origin = catalog[0].preferred_origin()
    picks = []
    for arrival in origin.arrivals:
        picks.append(arrival.get_pick())

    obs2 = nlloc.Observations(picks)

    assert str(obs) == str(obs2)


def test_write_velocity_grid_files(velocity_grids):
    velocity_grids.write(path=velocity_grid_file)
    p_velocity = grid.read_grid('TEST.P.mod', path=velocity_grid_file)
    assert np.all(p_velocity.data == velocity_grids['P'].data)


def test_write_travel_time_grids(travel_time_grids):
    travel_time_grids.write(travel_time_root)
    base_name = travel_time_grids[0].base_name
    travel_time_grid = grid.read_grid(base_name, path=travel_time_root)
    assert np.all(travel_time_grid.data == travel_time_grids[0].data)


def test_build_nlloc_control_file(inventory):
    assert True


if __name__ == '__main__':
    pass
    # write_velocity_grids()
    # write_travel_time_grids()





