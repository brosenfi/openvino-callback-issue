#!/bin/bash

echo "First argument is $1"

if [ "$#" -ne 1 ]; then
    echo "Please provide a single argument that is the number of test requests (integer)"
    exit 1
fi

export TEST_NUM_REQUESTS="$1"
echo "Number of requests is ${TEST_NUM_REQUESTS}"

source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh

python3 test_vgg16_example.py

exit 0
