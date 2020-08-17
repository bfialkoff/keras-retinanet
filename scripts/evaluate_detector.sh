#!/bin/bash
. ${BASH_SOURCE%/*}/vars.sh

date=202008171624
weight_save_path=${storage_dir}${date}/weights/


score_thresh=0.4
max_detections=25
should_draw=0


for e in $(seq 36 1 36)
do
epoch=`printf %02d $e`
model=${weight_save_path}${back_bone}_csv_${epoch}.h5
echo $model
save_path=${storage_dir}${date}/eval/epoch_${epoch}/

python ${eval_script} --max-detections ${max_detections} --score-threshold ${score_thresh} \
--backbone ${back_bone} --save-path ${save_path} csv ${not_train_ann} ${class_map} ${model} --convert-model
#  --config ${anchor_path}
done
