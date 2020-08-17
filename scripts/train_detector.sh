#!/bin/bash
. ${BASH_SOURCE%/*}/vars.sh

weight_save_path=${storage_dir}${date}/weights/
tensorboard_path=${storage_dir}${date}/tensorboard/

initial_epoch=0
b_size=4
steps=37
epochs=500

echo TF_CUDNN_USE_AUTOTUNE=0 python3 ${train_script} --backbone ${back_bone} \
--epochs $epochs --batch-size $b_size --steps $steps --initial-epoch ${initial_epoch} \
--snapshot-path ${weight_save_path} --compute-val-loss \
--tensorboard-dir ${tensorboard_path} \
csv ${train_ann} ${class_map} --val-annotations ${val_ann}

#--config ${anchor_path} \

