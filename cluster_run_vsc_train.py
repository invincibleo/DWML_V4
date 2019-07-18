#!/usr/bin/env python
# @Time    : 09/04/2019
# @Author  : invincibleo
# @Site    : https://iiw.kuleuven.be/onderzoek/emedia/people/phd-students/duoweitang
# @File    : cluster_run_vsc
# @Software: PyCharm

import os
import time
import tensorflow as tf

if __name__ == "__main__":
    flags = tf.flags
    # Learning related FLAGS
    flags.DEFINE_boolean("is_training", False, "True for training, False for testing [Train]")
    FLAGS = flags.FLAGS

    seq_length = 500
    model = "e2e_2018_seperateAE"
    data_dir = '$VSC_DATA/datasets/tf_records'
    log_dir = '$VSC_DATA/Learning_outputs/stateOfTheArtReproduce/{}'.format(model)
    head = ["#!/bin/bash\n",
            "#PBS -M duowei.tang@kuleuven.be -m abe\n",
            "#PBS -r y\n",
            "#PBS -j oe\n",
            "echo \"Start - `date`\"\n",
            "cd DWML_V4/\n",
            "source /data/leuven/324/vsc32400/miniconda3/bin/activate py3\n",
            "echo \"Program outputs begins:\"\n"]
    command = "python train.py" \
              " --data_dir=%s" \
              " --output_dir=%s" \
              " --model=%s" \
              " --seq_length=%d" \
              " --batch_size=2" \
              " --learning_rate=0.0001" \
              " --is_learning_rate_decay=False" \
              % (data_dir,
                 log_dir,
                 model,
                 seq_length)

    print("Writing into bash file")
    with open('bash_tmp.sh', 'w') as f:
        f.writelines(head)
        f.writelines(command)
    time.sleep(1)
    run_qsub = "qsub -l partition=gpu " \
               "-l nodes=1:ppn=9:gpus=1 " \
               "-l walltime=72:00:00 " \
               "-A lp_stadius_dsp bash_tmp.sh"
    os.system(run_qsub)
    print("Running...")
    time.sleep(1)

