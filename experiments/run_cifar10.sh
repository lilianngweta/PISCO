#!/bin/bash
for job in {0..9599}
do
    echo "Running job ${job}."
    python3 experiments_cifar10.py ${job}
done

