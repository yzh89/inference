#!/bin/bash
# Run script for OSS SSD

set -e

CLOUD_TPU="cloud_tpu"
DATA_DIR="../coco"
MODEL_DIR="../pretrained/ssd_test"
# MODEL_DIR="gs://garden-team-scripts/ssd_test_model_dir"


# TODO(taylorrobie): properly open source backbone checkpoint
RESNET_CHECKPOINT="../pretrained/ssd-resnet-checkpoint"

# chmod +x open_source/.setup_env.sh
# ./open_source/.setup_env.sh

# if [ ! -d $DATA_DIR ]; then
#   pushd ${CLOUD_TPU}/tools/datasets
#   bash download_and_preprocess_coco.sh $DATA_DIR
#   popd
# fi

# if [ -d $MODEL_DIR ]; then
#   rm -r $MODEL_DIR
# fi

python3 ssd_main.py  --use_tpu=False \
                     --device=gpu \
                     --mode=train \
                     --train_batch_size=4 \
                     --training_file_pattern="${DATA_DIR}/train-*" \
                     --resnet_checkpoint=${RESNET_CHECKPOINT} \
                     --model_dir=${MODEL_DIR} \
                     --num_epochs=10

python3 ssd_main.py  --use_tpu=False \
                     --device=gpu \
                     --mode=eval \
                     --eval_batch_size=4 \
                     --validation_file_pattern="${DATA_DIR}/val-*" \
                     --val_json_file="${DATA_DIR}/annotations/instances_val2017.json" \
                     --model_dir=${MODEL_DIR}\
                     --eval_timeout=0
