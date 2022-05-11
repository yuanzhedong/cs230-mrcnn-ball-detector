#!/bin/bash

set -x
CONT=mrcnn
docker build -t $CONT .

docker run --rm -ti --net=host --ipc=host --gpus all --privileged \
    --volume="${PWD}":"/workspace" \
    --name mrcnn_training \
    $CONT \
    bash
