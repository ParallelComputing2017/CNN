#!/bin/bash

threads=14
mode="single"
cnn="./Release/CNN"

# Sequential version
eval "$cnn $mode $threads"

# Parallel version
mode="all"

for t in {1..16}
do
	threads=$t
	eval "$cnn $mode $threads"
	printf "\n"
done
