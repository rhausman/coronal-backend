## Setup
Run 
```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install poetry
    poetry install
```

To add a dependency: `pip add [dependency name]`


# API
To build updated version, use `docker-compose up -d --build`
To run, use `docker-compose up`
or, to run without a docker container, use `uvicorn api:app --reload --workers 1 --host 0.0.0.0 --port 8000`