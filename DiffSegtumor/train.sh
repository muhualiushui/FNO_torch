#!/bin/bash

while getopts 'e:c:i:l:w:t:n:d:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    i) identifier=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) stu_w=$OPTARG;;
		    t) task=$OPTARG;;
		    n) num_epochs=$OPTARG;;
		    d) train_flag=$OPTARG;;
    esac
done
echo "exp:" $exp
echo "cuda:" $cuda
echo "num_epochs:" $num_epochs
echo "train_flag:" $train_flag




if [ ${task} = "brats_2d" ];
  then
    labeled_data="train_merged_t2"
    unlabeled_data="train_merged_t2"
    eval_data="eval_merged_t2"
    test_data="eval_merged_t2"
    modality="MR"
    folder="Exp_tumorseg_brats/"
    if [ ${train_flag} = "true" ]; then
      python code/train_diffusion_2d.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    # python code/test_2d.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    # python code/evaluate_2d.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} --modality ${modality} -t ${task}
fi



if [ ${task} = "brats_diff" ];
  then
    labeled_data="train_merged_t2"
    unlabeled_data="train_merged_t2"
    eval_data="eval_merged_t2"
    test_data="eval_merged_t2"
    modality="MR"
    folder="Exp_tumorseg_bratsdiff/"
    if [ ${train_flag} = "true" ]; then
      python code/train_diffusion_diff.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    # python code/test_2d.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    # python code/evaluate_2d.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} --modality ${modality} -t ${task}
fi


# bash train.sh -c 0 -e diffusion -t brats_2d -i '' -l 1e-2 -w 10 -n 300 -d true 
# bash train.sh -c 2 -e diffusion -t brats_diff -i '' -l 1e-2 -w 10 -n 300 -d true 