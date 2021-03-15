import uvicorn

if __name__ == '__main__':
    uvicorn.run('grid:app', reload=True)
