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

    is_learning_rate_decay_list = [False]
    seq_length_list = [500]
    test_seq_length_list = [250, 300, 500, 750, 1875, 3750]
    dropout_list = [0.0]
    l2_list = [0.0001]
    input_noise_std_list = [0.01]
    data_dir = '$VSC_DATA/datasets/RECOLA_16K_old'
    model = "e2e_2018_provide"

    for i in range(5):
        for seq_length in seq_length_list:
            for is_learning_rate_decay in is_learning_rate_decay_list:
                for l2 in l2_list:
                    for dropout in dropout_list:
                        for input_noise_std in input_noise_std_list:
                            for test_seq_length in test_seq_length_list:
                                log_dir = '$VSC_DATA/Learning_outputs/stateOfTheArtReproduce/{}_{}_seq_{}_RMSProp_bs5'.format(i, model, seq_length)

                                head = ["#!/bin/bash\n",
                                        "echo \"Start - `date`\"\n",
                                        "cd DWML_V6/\n",
                                        "source /data/leuven/324/vsc32400/miniconda3/bin/activate py3\n",
                                        "echo \"Program outputs begins:\"\n"]
                                command = "python SOTA_testing.py" \
                                          " --data_dir=%s" \
                                          " --chk_file_dir=%s" \
                                          " --learning_rate=0.0001" \
                                          " --model=wavenet" \
                                          " --dataset=AVEC2016_RECOLA" \
                                          " --seq_length=%d" \
                                          " --batch_size=100" \
                                          " --lc_channels=0" \
                                          " --is_learning_rate_decay=%s" \
                                          " --checkpoint_num=-1" \
                                          " --l2_regularization_strength=%f" \
                                          " --dropout_m=%f" \
                                          % (data_dir,
                                             log_dir,
                                             test_seq_length,
                                             is_learning_rate_decay,
                                             l2,
                                             dropout)

                                print("Writing into bash file")
                                with open('bash_tmp.sh', 'w') as f:
                                    f.writelines(head)
                                    f.writelines(command)
                                time.sleep(1)
                                run_qsub = "qsub -l partition=gpu " \
                                           "-l nodes=1:ppn=9:gpus=1 " \
                                           "-l walltime=00:10:00 " \
                                           "-A lp_stadius_dsp bash_tmp.sh"
                                os.system(run_qsub)
                                print("Running...")
                                time.sleep(1)
