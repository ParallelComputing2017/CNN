#!/bin/bash

threads=0
mode="single"
cnn="./Release/CNN"

# Sequential version
eval "$cnn $mode $threads"

# Parallel version
mode="all"

for t in {1..8}
do
	threads=$t
	eval "$cnn $mode $threads"
	printf "\n"
done
