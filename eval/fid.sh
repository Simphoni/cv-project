TRUTH="/home/ubuntu/inference/data/test_top256"
DATA="/home/ubuntu/inference/output_test/lora-rank128-snr-bs6-encoder-sample500"

python pytorch-fid/src/pytorch_fid/fid_score.py $TRUTH $DATA --device cuda

#!/bin/bash

# TRUTH="/home/ubuntu/inference/data/test_top256"
# OUTPUT_TEST="output_test"

# # 遍历 output_test 文件夹下的所有子目录
# for DATA_DIR in ${OUTPUT_TEST}/*; do
#     # 检查是否为目录
#     if [ -d "$DATA_DIR" ]; then
#         echo "Processing folder: $DATA_DIR"

#         # 运行 FID 分数计算
#         python pytorch-fid/src/pytorch_fid/fid_score.py "$TRUTH" "$DATA_DIR" --device cuda

#         echo "Finished processing folder: $DATA_DIR"
#         echo "-----------------------------------"
#     fi
# done

# TRUTH="/home/ubuntu/inference/data/train_top500"
# OUTPUT_TEST="output"

# # 遍历 output_test 文件夹下的所有子目录
# for DATA_DIR in ${OUTPUT_TEST}/*; do
#     # 检查是否为目录
#     if [ -d "$DATA_DIR" ]; then
#         echo "Processing folder: $DATA_DIR"

#         # 运行 FID 分数计算
#         python pytorch-fid/src/pytorch_fid/fid_score.py "$TRUTH" "$DATA_DIR" --device cuda

#         echo "Finished processing folder: $DATA_DIR"
#         echo "-----------------------------------"
#     fi
# done
