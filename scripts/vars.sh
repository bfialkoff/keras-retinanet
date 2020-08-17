#!/bin/bash
date=`/bin/date +%Y%m%d%H%M`
user=`whoami`

storage_dir=/media/adam/e46d6141-876f-4b0c-90da-9e9e217986f2/betzalel_personal/araplus/
base_dir=/home/${user}/Desktop/betzalel_personal/keras-retinanet/
annotations_dir=${base_dir}data/araplus/annotations/

annotations_file=${annotations_dir}unsplit.csv
train_ann=${annotations_dir}train_annotations.csv
val_ann=${annotations_dir}val_annotations.csv
test_ann=${annotations_dir}val_annotations.csv
not_train_ann=${annotations_dir}not_train_annotations.csv
class_map=${annotations_dir}class_map.csv
anchor_path=${annotations_dir}config.ini

back_bone="resnet50"

train_script=${base_dir}keras_retinanet/bin/train.py
eval_script=${base_dir}keras_retinanet/bin/evaluate.py
optimization_script=${base_path}keras_retinanet/bin/optimize_anchors.py
debug_script=${base_dir}keras_retinanet/bin/debug.py
convert_script=${base_dir}keras_retinanet/bin/convert_model.py