FROM python:3.8-slim-buster

ENV VIRTUAL_ENV=/opt/venv
# make the venv, and append to the PATH
RUN python -m venv $VIRTUAL_ENV && pip install poetry
# source from the virtualenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
RUN poetry install

COPY . .