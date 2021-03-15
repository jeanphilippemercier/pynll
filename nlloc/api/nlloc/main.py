import uvicorn
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from nlloc.nlloc import ProjectManager
from pathlib import Path
import numpy as np
from nlloc.api.nlloc.settings import settings
print(Settings.Config.env_prefix)
from ipdb import set_trace
set_trace()

app = FastAPI()

nlloc_path = Path(settings.project_root)


@app.post('/create_project/{project_name}')
async def create_project():
    pm = ProjectManager(nlloc_path, 'TEST', 'TEST')
    pass


@app.get('/list_projects')
async def list_projects():
    return {'message': nlloc_path.glob('*')}


@app.get('/grids/travel_time/{sensor}')
async def get_travel_time(sensor: str, x: float, y: float, z: float,
                          phase: List[str] = Query(['P', 'S']),
                          grid_coordinates: Optional[bool] = False):
    # set_trace()
    tt_grids = pm.travel_times.select(seed_labels=sensor, phase=phase)
    loc = np.array([x, y, z])

    # test if the point is inside the grid
    grid_shape = np.array(tt_grids[0].shape)
    if grid_coordinates:
        in_grid = np.all(loc < grid_shape)
    else:
        in_grid = tt_grids[0].in_grid(loc)

    if not in_grid:
        message = f'point {loc} is outside the grid'
        raise HTTPException(status_code=404, detail=message)

    # for tt_grid in tt_grids:
    travel_time = tt_grids[0].interpolate(loc,
                                          grid_coordinate=grid_coordinates)[0]
    # set_trace()
    # from ipdb import set_trace; set_trace()

    print(travel_time)
    return {'message': str(travel_time)}


@app.get('/inventory/sensors')
async def get_sensor_list():
    sensor_codes = [sensor.code for sensor in pm.inventory.sensors]
    return {'message': sensor_codes}


if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)

