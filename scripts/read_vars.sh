#!/bin/basg

. ${BASH_SOURCE%/*}/vars.sh
echo "python ${debug_script} --config ${anchor_path} csv ${annotations_file} ${class_map}"
