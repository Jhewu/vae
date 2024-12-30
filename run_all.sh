#!/bin/bash

echo "starting run..."

python3 run_vae_3.py
python3 run_vae.py
python3 run_vae_2.py
python3 run_vae_4.py

echo "finished running..."