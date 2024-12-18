TEXT="/home/ubuntu/inference/data/test_text"
IMAGE="/home/ubuntu/inference/data/test_top256"

python clip-score/src/clip_score/clip_score.py $IMAGE $TEXT --device "cuda"

# TEXT="/home/ubuntu/inference/data/test_text"
# OUTPUT_TEST="output_test"

# # 遍历 output_test 文件夹下的所有子目录
# for DATA_DIR in ${OUTPUT_TEST}/*; do
#     # 检查是否为目录
#     if [ -d "$DATA_DIR" ]; then
#         echo "Processing folder: $DATA_DIR"

#         # 运行 FID 分数计算
#         python clip-score/src/clip_score/clip_score.py "$DATA_DIR" "$TEXT" --device cuda

#         echo "Finished processing folder: $DATA_DIR"
#         echo "-----------------------------------"
#     fi
# done