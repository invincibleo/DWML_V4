#!/bin/bash
#PBS -M duowei.tang@kuleuven.be -m abe
#PBS -r y
#PBS -j oe
echo "Start - `date`"
cd DWML_V4
source /data/leuven/324/vsc32400/miniconda3/bin/activate py3new
echo
echo "Program outputs begins:"
python AE_train.py --seq_length 500 --batch_size 10 --learning_rate_decay false --learning_rate 0.0001 --output_dir e2e_2018_AE
echo "End - `date`"

