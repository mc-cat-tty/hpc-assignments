#!/bin/bash

FILENAME=$1
gcc -fopenmp -E -I. -I../utilities $FILENAME
