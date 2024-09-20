# script for applying mask

python /path/to/main.py --mask text_vqa_5000_0.01 \
                        --mask_modules vit llm \
                        --save_output 1
                        

python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask spe_text --gpu 0
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask image --gpu 1
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask special --gpu 2
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask spe_text image --gpu 3
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask text special --gpu 4
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask text special image --gpu 5
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask special image --gpu 6

python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask spe_text --gpu 0
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask image --gpu 1
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask spe_text image --gpu 3
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection uniform --modality_mode 1 --select_to_mask spe_text --gpu 2
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection uniform --modality_mode 1 --select_to_mask image --gpu 4



python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 0 --selection uniform --select_ratio 0.01
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 1 --selection adaptive --select_ratio 0.01
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 2 --selection LU_MA --select_ratio 0.01
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 3 --selection LA_MU --select_ratio 0.01
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 4 --selection uniform --select_ratio 0.02
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 5 --selection adaptive --select_ratio 0.02
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 6 --selection LU_MA --select_ratio 0.02
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 7 --selection LA_MU --select_ratio 0.02

python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 0 --selection uniform --select_ratio 0.015
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 1 --selection adaptive --select_ratio 0.015
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 2 --selection LU_MA --select_ratio 0.015
python /mnt/kaichen/modality_specific/main.py --mode 2 --modality_mode 1 --gpu 3 --selection LA_MU --select_ratio 0.015

# 改变ratio（0.01,0.02）和selection来测试selection效果





python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask image --gpu 1 --select_ratio 0.015

# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask image --gpu 1 --select_ratio 0.15
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection uniform --modality_mode 2 --select_to_mask image --gpu 3 --select_ratio 0.015
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection uniform --modality_mode 2 --select_to_mask image --gpu 3 --select_ratio 0.15

python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask image --gpu 5 --select_ratio 0.02
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection adaptive --modality_mode 1 --select_to_mask image --gpu 5 --select_ratio 0.2
python /mnt/kaichen/modality_specific/main.py --mode 2 --selection uniform --modality_mode 2 --select_to_mask image --gpu 7 --select_ratio 0.02
# python /mnt/kaichen/modality_specific/main.py --mode 2 --selection uniform --modality_mode 2 --select_to_mask image --gpu 7 --select_ratio 0.2


python /mnt/kaichen/modality_specific/main.py --mode 2 --selection layer_uniform --modality_mode 2 --select_to_mask text special image --gpu 6 --select_ratio 0.015


python main.py --dataset coco_caption --mode 1 --sample_num -1 --gpu 0
python main.py --dataset text_vqa --mode 1 --sample_num -1 --gpu 1
python main.py --dataset mmlu --mode 1 --sample_num -1 --gpu 2