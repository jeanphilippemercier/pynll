import uvicorn
from typing import Optional, List
from fastapi import FastAPI, Query
from ..nlloc import ProjectManager
import numpy as np

app = FastAPI()

@app.get('grids/traveltime/{sensor}'):
async def get_travel_time(sensor: str, x: float, y: float, z: float,
                          phase: List[str] = Query(['P', 'S'],
                                                   title=''),
                          grid_coordinates: Optional[bool] = True):
    nlloc_path = '/home/jpmercier/Repositories/uquake-project/pynll/tests'
    pm = ProjectManager(nlloc_path, 'TEST', 'TEST')
    tt = pm.travel_times.select(seed_labels=sensor)
    loc = np.array(x, y, z)
    travel_time = tt[0].interpolate()


if __name__ == '__main__':
    uvicorn.run('nlloc_main:app', reload=True)

