#!/bin/bash
for job in {0..3599}
do
    echo "Running job ${job}."
    python3 expt.py ${job}
done
