# Evaluating from folder
python large_scale_evaluation.py \
    --dir example_images \
    --output-dir results/experiment_1 \
    --batch-size 4 \
    --num-noise 8 \
    --compute-stats

# Evaluating from csv
# python large_scale_evaluation.py \
#     --csv dataset.csv \
#     --image-col "image_path" \
#     --prompt-col "caption" \
#     --output-dir results/csv_experiment \
#     --batch-size 8

# Comparing different datasets 
# python large_scale_evaluation.py \
#     --multi-dir real_images/ generated_sd/ generated_dalle/ \
#     --labels "Real" "StableDiffusion" "DALLE" \
#     --output-dir results/comparison \
#     --batch-size 4 \
#     --compute-stats

# Evaluating from image list
# python large_scale_evaluation.py \
#     --txt image_list.txt \
#     --output-dir results/txt_experiment \
#     --find-outliers