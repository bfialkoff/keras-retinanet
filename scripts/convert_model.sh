#!/bin/bash
. ${BASH_SOURCE%/*}/vars.sh

date=202008171624

python ${convert_script} ${storage_dir}${date}/weights/${back_bone}_csv_36.h5 \
${storage_dir}202008171624_pred_model_resnet50_csv_36.h5



