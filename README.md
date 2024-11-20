# Execution Guide

## Overview
---
Script location: `./Results/run.bash` 

It is used to compile and run different implementations (`Sequential`, `OpenMP`, `OpenCilk`, and `Pthreads`) on specified datasets with configurable parameters.

---

## How to Execute

### Basic Command
```bash
bash run_script.sh <type> <dataset> <sampling_reduction> <candidate_reduction> [MIN_SIZE_factor] [THREADS_NUM]
