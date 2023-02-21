#!/bin/bash
for job in {0..239}
do
    echo "Running job ${job}."
    python3 experiments_mnist.py ${job}
done


