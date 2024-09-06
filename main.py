"""main python file"""
import os
import json
import argparse
import pickle
import sys
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(description="A simple example script to demonstrate argparse.")    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "one_llm"], help="the mllm to be analyzed")
    # choices of all_modules can be augmented if needed
    parser.add_argument('--all_modules', type=str, choices=["vit", "llm"], nargs='+', default=["vit", "llm"], help="all modules appear in current MLLM")
    parser.add_argument('--gpu', type=str, default='0', help="gpu_id")

    parser.add_argument('--task', type=str, default='text_vqa', choices=["test", "text_vqa", "coco_caption"], help="type of qa datasets")
    parser.add_argument('--vqa_num', type=int, default=1, help="number of vqa questions, -1 means all questions")
    parser.add_argument('--save_output', type=int, default=0, help="whether to save outputs in txt file")
    
    # choices of args.mask_modules must keep in same as all_modules
    parser.add_argument('--mask_modules', type=str, choices=["vit", "llm"], nargs='+', default=None, help="modules we choose to mask")
    parser.add_argument('--mask', type=str, default=None, help="mask file need to load, such as test_1_0.01")
    
    parser.add_argument('--save_mask', type=int, default=0, help="whether to save mask?")
    parser.add_argument('--select_ratio', type=float, default=0.01, help="the ratio of selected neurons")

    args = parser.parse_args()
    # gpu id must be set before "import torch"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from utils import get_top_acts, check_acc, set_seed, safe_infer, karpathy_coco_test, TextEvaluation, split_mask_prefix
    from models import Qwen2_VL
    set_seed(2024)

    # save outputs txt w.o. mask or with certain type of mask
    if args.save_output:
        mask_flag = f'mask_{"_".join(args.mask_modules)}_{args.mask}' if args.mask_modules is not None else 'origin' 
        file_path = f'outputs/{args.mllm}/{args.task}/{args.vqa_num}_{mask_flag}.txt'
        if os.path.exists(file_path):
            print('work file exists! switch to next task!')
            return
        sys.stdout = open(file_path, 'w', encoding='utf-8')
    evaluator = TextEvaluation()

    # load mask
    if args.mask_modules is not None:
        tasks = ["test", "text_vqa", "coco_caption"] # keep same with "chioces" of args.task
        p1, p2 = split_mask_prefix(tasks, args.mask)
        mask_path = f'masks/{args.mllm}/{p1}/{p2}.pkl'
        print(f'load mask from {mask_path} and apply mask of {args.mask_modules}')
        with open(mask_path, 'rb') as f:
            masks = pickle.load(f)
        mask_modules = args.mask_modules
    else:
        masks = None
        mask_modules = None

    # create mllm model
    if args.mllm == 'qwen2_vl':
        model = Qwen2_VL(args, mask_modules, masks)
    else:
        pass # add more models

    # processing questions from various benchmarks
    if args.task == 'test':
        # data is a dict {'text': xxx, 'img': img_path, 'video': xxx, 'audio': xxx}
        data = {
            'text': "describe this image",
            'img': "scripts/test.jpg"
        }
        safe_infer(model, data)

    elif args.task == 'text_vqa':
        text_vqa_path = "/home/ubuntu/kaichen/data/TextVQA"
        with open(f"{text_vqa_path}/val/TextVQA_0.5.1_val.json", "r", encoding='utf-8') as f:
            text_vqa = json.load(f)['data']
        qa_num = len(text_vqa) if args.vqa_num == -1 else args.vqa_num
        correct_num = 0

        for i in range(qa_num):
            print(f'question {i}:')
            vqa = text_vqa[i]
            print(vqa)
            data = {
                'text': vqa['question'],
                'img': f"{text_vqa_path}/val/train_images/{vqa['image_id']}.jpg"
            }
            response = safe_infer(model, data)
            if check_acc(response, vqa['answers']):
                print('correct!')
                correct_num += 1
            else:
                print(f"wrong! answers are{vqa['answers']}")
        print(f'total {qa_num}, correct {correct_num}, accuracy {correct_num / qa_num}')

    elif args.task == 'coco_caption':
        coco_data = karpathy_coco_test()
        qa_num = len(coco_data) if args.vqa_num == -1 else args.vqa_num
        coco_prompt = """
                      Generate a caption for the image in one short sentence, similar to these examples from the COCO dataset: 
                      1. A man with a red helmet on a small moped on a dirt road.
                      2. Man riding a motor bike on a dirt road on the countryside.
                      3. A man riding on the back of a motorcycle.
                      4. A man in a red shirt and a red hat is on a motorcycle on a hill side.
                      Now, describe the image.
                      """
        coco_prompt = re.sub(r'\s+', ' ', coco_prompt.replace("\n", "").strip())

        scores = {'bleu': 0, 'clip_score': 0, 'sbert_similarity': 0, 'cider': 0}
        for index, (_, value) in enumerate(coco_data.items()):
            print(f'question {index}:')
            if index < qa_num:
                gt_captions = value['anns']
                print(f"img_path: {value['img']}")
                data = {
                    'text': coco_prompt,
                    'img': value['img']
                }
                pred_caption = safe_infer(model, data)
                print(f'gt_captions: {gt_captions}')
                results = evaluator.evaluate(pred_caption, gt_captions)
                for score_name in scores.keys():
                    scores[score_name] += results[score_name]['max']
                    print(f'{score_name}: {results[score_name]}')
            else:
                break
        print(f'after processing {qa_num} images in total, the average scores are:')

        for score_name, score_value in scores.items():
            print(f'average {score_name}: {score_value / qa_num}')

    # select neurons and create mask
    if args.save_mask:
        mask_dict = {}
        for module, info in model.module_info.items():
            if module in args.all_modules:
                activation_matrix = info['activation_matrix'] / info['token_num']
                mask = get_top_acts(activation_matrix, ratio=args.select_ratio)
                mask_dict[module] = mask

        with open(f'masks/{args.mllm}/{args.task}/{args.vqa_num}_{args.select_ratio}.pkl', 'wb') as file:
            pickle.dump(mask_dict, file)

if __name__ == "__main__":
    main()