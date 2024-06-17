#This is for the Pretrain with the SH file

#$ -l tmem=22G
#$ -pe gpu 1
#$ -l gpu=true,gpu_type=rtx4090
#$ -l h_rt=100:00:00
#$ -j y
#$ -N FA_SL_img_
#$ -S /bin/bash
#$ -R y

# activate the virtual env

source activate retfound
# Source file for CUDA11.0
# 24/02/23

source /share/apps/source_files/cuda/cuda-11.0.source

nvidia-smi

hostname
date

# enter the project path
cd /SAN/ioo/AlzeyeTempProjects/boxuanli/RETFound_MAE-main
#gpu=2,3
myEpoch=99
MODEL_CFPFA99_epochs="/SAN/ioo/AlzeyeTempProjects/yukuzhou/RETFound_MAE-main-RETFound-CFPFA/output_dir/checkpoint-${myEpoch}.pth"
DATA_PATH_IDRiD="/SAN/ioo/AlzeyeTempProjects/boxuanli/IDRiD_data/"
DATA_PATH_APTOS2019="/SAN/ioo/AlzeyeTempProjects/boxuanli/APTOS2019/"

for n_round in 1 2 3 4
do
TASK_DIR="./finetune_CFPFA_epo${myEpoch}_IDRiD_${n_round}/"
# running command
################################################
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48793 main_finetune.py \
    --batch_size 24 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ${DATA_PATH_IDRiD} \
    --task ${TASK_DIR} \
    --finetune ${MODEL_CFPFA99_epochs} \
    --input_size 224 \
    --seed ${n_round}
done
date


date
for n_round in 0 1 2 3 4
do

TASK_DIR="./external_CFPFA_epo${myEpoch}_IDRiD_${n_round}/"
RESUME_DIR="./finetune_CFPFA_epo${myEpoch}_IDRiD_${n_round}/checkpoint-best.pth"
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48793 external_validation.py \
    --eval \
    --batch_size 24 \
    --world_size 2 \
    --model vit_large_patch16 \
    --epochs 40 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ${DATA_PATH_IDRiD} \
    --task ${TASK_DIR} \
    --resume ${RESUME_DIR} \
    --seed ${n_round}
done
date
