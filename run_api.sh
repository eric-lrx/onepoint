#!/bin/bash

python3 -m venv ./api/virtenv
source ./api/virtenv/bin/activate
pip install -r api/requirements.txt
python3 api/main.py