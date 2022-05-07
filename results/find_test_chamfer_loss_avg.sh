#!/bin/bash

file=./test.txt

echo `awk '{total += $5;count++} END {print total/count}' $file` 

