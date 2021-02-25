from nlloc import nlloc
from importlib import reload
from .fixtures.data import inventory, catalog
from loguru import logger
from pathlib import Path

reload(nlloc)
project_dir = 'NLL'


def test_create_observations(catalog):
    obs = nlloc.Observations.from_event(catalog)

    with open('NLL/test.obs', 'w') as obs_file:
        obs_file.write(str(obs))

    origin = catalog[0].preferred_origin()
    picks = []
    for arrival in origin.arrivals:
        picks.append(arrival.get_pick())

    obs2 = nlloc.Observations(picks)

    assert str(obs) == str(obs2)


root_path = Path('NLL')
project = 'TEST'
observation_dir = 'obs'
travel_time_dir = 'time'
output_file_dir = 'hyp'

observation_files = root_path / observation_dir / 'test'
travel_time_dir = root_path / observation_dir

def test_write_travel_time_grids():
    assert True


def test_build_nlloc_control_file(inventory):
    control = nlloc.Control()
    geographic_transformation = nlloc.GeographicTransformation()
    sensors = nlloc.Sensors.from_inventory(inventory)
    files = nlloc.InputFiles()






