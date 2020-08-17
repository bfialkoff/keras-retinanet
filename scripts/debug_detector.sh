#!/bin/bash
. ${BASH_SOURCE%/*}/vars.sh

save_dir=${base_dir}images/debug/araplus/

python ${debug_script} --config ${anchor_path} csv ${annotations_file} ${class_map}



