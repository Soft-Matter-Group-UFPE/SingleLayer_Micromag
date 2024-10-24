#!/bin/bash
mkdir ./mx
mkdir ./my
mkdir ./mz
mkdir ./m_final
mkdir ./Figures_Mag
ulimit -s 100000
gcc -fopenmp SingleLayer_SK.c -o ./Program.out -lm -O3 -mavx2 -mtune=native -march=native 
./Program.out
