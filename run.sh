#!/bin/bash
#PBS -M duowei.tang@kuleuven.be -m abe
#PBS -r y
#PBS -j oe
echo "Start - `date`"
cd DWML_V4
source /data/leuven/324/vsc32400/miniconda3/bin/activate py3new
echo
echo "Program outputs begins:"
python AE_train.py --seq_length 500 --batch_size 10 --epochs 300 --learning_rate_decay false --learning_rate 0.0001 --latent_dim 256 --output_dir $VSC_DATA/Learning_outputs/e2e_2018_AE_dense_batchnorm --dataset_dir $VSC_DATA/datasets/tf_records
echo "End - `date`"

