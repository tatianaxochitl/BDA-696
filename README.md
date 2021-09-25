# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# Run Assignment-1

- Do after following instructions for Setup for development
- You can run assignment from current directory `python3 assignment1.py`

# Assignment 2 Instructions

1. Make sure that Baseball database is set up (not altered if altered edit out line 2 of assignment-1.sql )
2. Sign into database `mysql -u [username] -p [baseball database]`
3. type `source assignement-2.sql` (note: takes a while to run the file...)
