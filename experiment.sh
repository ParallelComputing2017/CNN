#!/bin/bash


threads=0
mode="sequential"
cnn="./Release/CNN"

# Sequential version
eval "$cnn $mode $threads"

# Parallel version
cpu_modes=("pthreads" "openmp")

for mode in ${cpu_modes[*]}
do
	for t in {1..8}
	do
		threads=$t
		#echo $mode
		eval "$cnn $mode $threads"
		printf "\n"
	done
done

gpu_modes=("cuda" "opencl")

for mode in ${gpu_modes[*]}
do
	for t in {1..1}
	do
		threads=$t
		#echo $mode
		eval "$cnn $mode $threads"
		printf "\n"
	done
done
