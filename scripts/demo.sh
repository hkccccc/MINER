# switch args.demo to 1 before run the demo program
# conda activate radar

rm -rf demo_outputs

# infer mormally
python main.py --dataset test --mode 0 --sample_num 1 --demo 1

# stage1, generate ISM
python main.py --dataset coco_caption --mode 1 --sample_num 22 --demo 1
python main.py --dataset mmlu --mode 1 --sample_num 26 --demo 1
python main.py --dataset text_vqa --mode 1 --sample_num 29 --demo 1

# stage1.5
python main.py --mllm qwen2_vl --mode 1.5 --load_ISM_sample_num -1 -1 -1 -1 -1 -1 --demo 1

# stage2 (can be skipped)
python main.py --mllm qwen2_vl --mode 2 --demo 1

# stage3
python main.py --mllm qwen2_vl --mode 3 --demo 1 --dataset mmlu --sample_num 22 --sum_ISM_path text_vqa29_coco_caption22_mmlu26