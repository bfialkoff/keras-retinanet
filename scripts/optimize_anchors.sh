#!/bin/bash
. ${BASH_SOURCE%/*}/vars.sh

num_ratios=3
num_scales=3

python ${optimization_script} ${annotations_file} --ratio ${num_ratios} --scales ${num_scales}
