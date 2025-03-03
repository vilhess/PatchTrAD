#!/bin/bash

# Define datasets to iterate over
datasets=("nyc_taxi" "ec2_request_latency_system_failure" "msl" "swat" "smap" "smd")

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"
    python main.py dataset=$DATASET
done
