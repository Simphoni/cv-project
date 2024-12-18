# GROUND_TRUTH="/home/ubuntu/inference/data/test_top256"
# IMG_DIR="/home/ubuntu/inference/output_test/pure_model"

# python PerceptualSimilarity/lpips_2dirs.py \
#     -d0 $IMG_DIR -d1 $GROUND_TRUTH -o ${IMG_DIR}/dists.txt --use_gpu

OUTPUT_TEST="output_test"
GROUND_TRUTH="/home/ubuntu/inference/data/test_top256"

# 遍历 output_test 文件夹下的所有子目录
for DATA_DIR in ${OUTPUT_TEST}/*; do
    # 检查是否为目录
    if [ -d "$DATA_DIR" ]; then
        echo "Processing folder: $DATA_DIR"

        # 运行 FID 分数计算
        python PerceptualSimilarity/lpips_2dirs.py -d0 $DATA_DIR -d1 $GROUND_TRUTH -o ${DATA_DIR}/dists.txt --use_gpu

        echo "Finished processing folder: $DATA_DIR"
        echo "-----------------------------------"
    fi
done