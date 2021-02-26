from uquake.core.inventory import read_inventory
from uquake.core.event import read_events
from uquake.core.stream import read
from pytest import fixture
import os
from time import time

test_artifacts = os.environ['UQUAKE_TEST_ARTIFACTS']


def get_inventory():
    inventory_file = test_artifacts + '/inventory.xml'
    inventory = read_inventory(inventory_file)
    return inventory


@fixture
def inventory():
    return get_inventory()


def get_catalog():
    event_file = test_artifacts + '/event_file.xml'
    inventory = get_inventory()
    cat = read_events(event_file)
    for i, pick in enumerate(cat[0].picks):
        for sensor in inventory.sensors:
            if sensor.alternate_code == pick.waveform_id.station_code:
                cat[0].picks[i].waveform_id.network_code = inventory[0].code
                cat[0].picks[i].waveform_id.station_code = sensor.station.code
                cat[0].picks[i].waveform_id.location_code = \
                    sensor.location_code
                cat[0].picks[i].waveform_id.channel_code = \
                    sensor.channels[0].code
                break
    return cat


@fixture
def catalog(inventory):
    return get_catalog()


def get_waveform():
    waveform_file = test_artifacts + '/waveform.mseed'
    st = read(waveform_file)
    for i, tr in enumerate(st):
        for sensor in inventory.sensors:
            if sensor.alternate_code == tr.stats.station:
                st[i].stats.network = inventory[0].code
                st[i].stats.station = sensor.station.code
                st[i].stats.location = sensor.location_code
                for channel in sensor.channels:
                    if tr.stats.channel in channel.code:
                        st[i].stats.channel = channel.code
                        break
                break
    return st


@fixture
def waveform(inventory):
    return get_waveform()


@fixture
def travel_time_grids():
    t0 = time()
    travel_time_grids_s = vel_3d_s.to_time_multi_threaded(seeds, seed_labels)
    travel_time_grids_p = vel_3d_s.to_time_multi_threaded(seeds, seed_labels)
    t1 = time()
    logger.info(f'done calculating the travel time grids in {t1-t0:0.2f} '
                f'second')

    assert True
    return travel_time_grids_p + travel_time_grids_s