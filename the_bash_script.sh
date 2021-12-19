#!/bin/bash 
python3 -m pip install --upgrade pip

apt-get update\
    && apt-get install -y libmariadb3 libmariadb-dev\

python3 -m pip install -r requirements.txt

mysql -u root -p root baseball < assignment-5.sql

python3 assignment-5.py

