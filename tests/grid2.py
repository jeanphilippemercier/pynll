from time import time
from loguru import logger
from .fixtures.grids import travel_time_grids
import numpy as np


def test_ray_tracing(travel_time_grids):
    t0 = time()
    origin = travel_time_grids[0].origin
    rays = travel_time_grids.ray_tracer(np.array(origin) + 200)
    t1 = time()
    logger.info(f'done calculating the rays in {t1 - t0:0.2f} second')
    assert True
    return rays


# if __name__ == '__main__':
#     travel_time_grids = test_create_travel_time()
#     test_ray_tracing(travel_time_grids)


