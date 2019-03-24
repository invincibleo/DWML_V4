#!/bin/bash
#PBS -M duowei.tang@kuleuven.be -m abe
#PBS -r y
#PBS -j oe
echo "Start - `date`"
cd DWML_V4/
source /data/leuven/324/vsc32400/miniconda3/bin/activate py3new
echo
echo "Program outputs begins:"
python siamese_train.py \
--batch_size 6 \
--epochs 5000 \
--seq_length 100 \
--latent_dim 256 \
--learning_rate_decay true \
--learning_rate 0.001 \
--output_dir $VSC_DATA/Learning_outputs/SER_out_siamese_LSTM \
--dataset_dir $VSC_DATA/datasets/tf_records
echo "End - `date`"

