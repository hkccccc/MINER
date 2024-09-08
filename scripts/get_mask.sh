# script for generating mask

vqa_num_list=(1 10 100 500 1000 2500 5000)
select_ratio_list=(0.0001 0.001 0.01 0.1)

for num in "${vqa_num_list[@]}"; do
    for ratio in "${select_ratio_list[@]}"; do        
        python path/to/main.py \
                          --vqa_num "$num" \
                          --select_ratio "$ratio" \
                          --save_mask 1 \
                          --gpu 0
    done
done