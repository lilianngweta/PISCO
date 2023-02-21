#!/bin/bash
for job in {0..59}
do
    echo "Running job ${job}."
    python3 experiments_imagenet_resnet50.py ${job}
done


