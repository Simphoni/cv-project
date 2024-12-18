DATA=/home/ubuntu/inference/data/test_top256

python inception-score-pytorch/inception_score.py $DATA

# OUTPUT_TEST="output_test"

# # 遍历 output_test 文件夹下的所有子目录
# for DATA_DIR in ${OUTPUT_TEST}/*; do
#     # 检查是否为目录
#     if [ -d "$DATA_DIR" ]; then
#         echo "Processing folder: $DATA_DIR"

#         # 运行 FID 分数计算
#         python inception-score-pytorch/inception_score.py "$DATA_DIR"

#         echo "Finished processing folder: $DATA_DIR"
#         echo "-----------------------------------"
#     fi
# done