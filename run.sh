#!/bin/bash

spark-submit \
    --master local[4] \
    --deploy-mode client \
    --queue default \
    --driver-memory 512M \
    --executor-memory 512M \
    --num-executors 3 \
/home/sasaki/dev/gamease/example/src/main/python/trainer.py 0
