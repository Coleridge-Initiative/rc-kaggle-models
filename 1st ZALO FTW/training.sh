#!/bin/sh

model_name="biomed_roberta_base scibert-base-cased"
n_exps=1 # number of experiments. We should train each single model
         # multiple times and get the best model based on the validation
         # score

for model in $model_name
do
    for i in $(seq 1 1 $n_exps)
    do
        mkdir -p ./saved_models/${model}/
        mkdir -p ./saved_models/${model}/run${i}

        python train.py training \
            --model_name ${model} \
            --exp_name run${i} \
            > ./saved_models/${model}/run${i}/log_training.txt 2>&1
    done
done
