"""main python file"""
import os
import json
import argparse
import sys
import re
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALL_MODALITIES = {
    'qwen2_vl': {
        'text_vqa': ["text", "special", "spe_text", "image", "all"],
        'coco_caption': ["text", "special", "spe_text", "image", "all"],
        'mmlu': ["text", "special", "all"]
    }
}
SELECTION_TYPES = ["uniform", "adaptive", "LA_MU", "LU_MA"]
ALL_TASKS = ["test", "text_vqa", "coco_caption", "mmlu"]

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(description="A simple example script to demonstrate argparse.")    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "one_llm"], help="the mllm to be analyzed")
    parser.add_argument('--task', type=str, default='text_vqa', choices=ALL_TASKS, help="type of qa datasets")
    parser.add_argument('--gpu', type=str, default='0', help="gpu_id")

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--mode', type=int, choices=[0,1,2], default=2)
    # mode 0: infer normally
    # mode 1: create folder and save mask
    # mode 2: load and apply mask

    # importance matrix
    parser.add_argument('--vqa_num', type=int, default=200, help="number of vqa questions, -1 means all questions")
    # get mask
    parser.add_argument('--ts', type=int, default=-1, help="timestep of importance matrix, -1 means last ts")
    parser.add_argument('--modality_mode', type=int, choices=[0,1,2,3], default=1, help="three types of mode")
    # mode 0: ['all']
    # mode 1: ['spe_text', ...]
    # mode 2: ['special', 'text', ...]
    # mode 3: ['random']
    parser.add_argument('--select_ratio', type=float, default=0.01, help="the ratio of selected neurons")
    parser.add_argument('--score_weights', type=float, nargs='+', default=[0,0.5,0.5,0,0], help="weights of [prob,mean,max,attn_k,attn_q], sum=1")
    parser.add_argument('--selection', type=str, choices=SELECTION_TYPES, default="layer_uniform")
    # apply mask
    parser.add_argument('--select_to_mask', type=str, nargs='+', default=None, help="select modals to mask")
    parser.add_argument('--deact_val', type=float, default=0, help="output value of a deactivated neuron, -1 means output.min()")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from utils import func as uf
    from models import Qwen2_VL

    uf.set_seed(2024)
    if args.vqa_num == -1:
        args.vqa_num = {'test': 1, 'text_vqa': 5000, 'coco_caption': 5000, 'mmlu': 14042}.get(args.task)
    
    args_dict = vars(args)
    cmd = [sys.executable] + sys.argv[:1]
    cmd += [f"--{key} {' '.join(map(str, value)) if isinstance(value, list) else value}" for key, value in args_dict.items() if value is not None]
    cmd_str = ' '.join(cmd)
    args.vqa_start_point = 0
    args.save_score_steps = [10, 100, 1000, 5000, 10000]
    
    all_modal = ALL_MODALITIES[args.mllm][args.task]
    modality_mode_map = {
        0: ['all'],
        1: [item for item in all_modal if item not in ['all', 'special', 'text']], # not for mmlu
        2: [item for item in all_modal if item not in ['all', 'spe_text']],
        3: ['random']
    }
    if args.mode == 1:
        args.mask_modal = all_modal
    elif args.mode == 2:
        args.mask_modal = modality_mode_map[args.modality_mode]
    
        # select_to_mask should be subset of mask_modal
        if args.select_to_mask is None:
            args.select_to_mask = args.mask_modal
        elif not set(args.select_to_mask).issubset(args.mask_modal):
            print("unsupported select_to_mask value...", f'{args.select_to_mask} not in {args.mask_modal}')
            sys.exit(0)
        else:
            args.select_to_mask = sorted(args.select_to_mask, key=lambda x: args.mask_modal.index(x))

    exp_name = '' if args.exp_name == '' else f'_{args.exp_name}'
    
    args.folder_path = f"outputs/{args.mllm}_{args.task}{exp_name}"
    csv_name = 'origin'
    if args.mode == 2:
        args.folder_path += f'/ts{args.ts}_mode{args.modality_mode}_r{args.select_ratio}_w{"_".join(map(str, args.score_weights))}_{args.selection}'
        csv_name = f'deact{args.deact_val}_mask_{"_".join(args.select_to_mask)}'
    
    if args.mode != 0:
        if not os.path.exists(args.folder_path):
            os.makedirs(args.folder_path)
            if args.mode == 1:
                os.makedirs(args.folder_path + 'importance_scores')
                with open(args.folder_path + '/run.sh', mode='w', encoding='utf-8') as f:
                    f.write(cmd_str)
                sys.stdout = open(args.folder_path + '/out.txt', 'a', encoding='utf-8')
            uf.initialize_csv(f'/{csv_name}.csv', args)
        
        elif args.mode == 2 and not os.path.exists(args.folder_path + f'/{csv_name}.csv'): # other deact_val
            uf.initialize_csv(f'/{csv_name}.csv', args)
        
        else:
            # check whether need to resume
            if args.mode == 1:
                sys.stdout = open(args.folder_path + '/out.txt', 'a', encoding='utf-8')
            df = pd.read_csv(args.folder_path + f'/{csv_name}.csv')

            if df['index'].iloc[-1] != args.vqa_num - 1:
                args.vqa_start_point = df['index'].iloc[-1] + 1
                args.mmlu_resume_args = (df["dataset name"].iloc[-1], df["sub-index"].iloc[-1])
                uf.initialize_csv(f'/{csv_name}.csv', args)
            else:
                print("Condition met, exiting the program...")
                sys.exit(0)

    evaluator = uf.TextEvaluation()

    # create mllm model
    if args.mllm == 'qwen2_vl':
        model = Qwen2_VL(args)
    else:
        pass # add more models

    # processing questions from various benchmarks
    if args.task == 'test':
        # data is a dict {'text': xxx, 'img': img_path, 'video': xxx, 'audio': xxx}
        data1 = {
            'text': "describe this image",
            'img': "scripts/test.jpg"
        }
        data2 = {
            'text': "What are the words in the picture",
            'img': "scripts/random.jpg"
        }
        data3 = {
            'text': 'what is the brand of this camera?',
            'img': '/home/ubuntu/kaichen/data/TextVQA/val/train_images/003a8ae2ef43b901.jpg'
        }
        response = model.infer(data1)
        print(response)

    elif args.task == 'text_vqa':
        text_vqa_path = "/mnt/kaichen/data/TextVQA"
        with open(f"{text_vqa_path}/val/TextVQA_0.5.1_val.json", "r", encoding='utf-8') as f:
            text_vqa = json.load(f)['data']

        for i in range(args.vqa_start_point, args.vqa_num):
            vqa = text_vqa[i]
            data = {
                'text': vqa['question'],
                'img': f"{text_vqa_path}/val/train_images/{vqa['image_id']}.jpg"
            }
            response = model.infer(data)
            cor = uf.check_acc(response, vqa['answers'])
            csv_line = {"index": i, "text": data["text"], "img": data["img"], 
                        "answer": response, "label": vqa['answers'], "correct": cor}
            uf.handle_output(args, csv_line)

    elif args.task == 'coco_caption':
        coco_data = uf.karpathy_coco_test()
        coco_prompt = """
                      Generate a caption for the image in one short sentence, similar to these examples from the COCO dataset: 
                      1. A man with a red helmet on a small moped on a dirt road.
                      2. Man riding a motor bike on a dirt road on the countryside.
                      3. A man riding on the back of a motorcycle.
                      4. A man in a red shirt and a red hat is on a motorcycle on a hill side.
                      Now, describe the image.
                      """
        coco_prompt = re.sub(r'\s+', ' ', coco_prompt.replace("\n", "").strip())
        for index, (_, value) in enumerate(coco_data.items(), start=args.vqa_start_point):
            if index >= args.vqa_num:
                break
            gt_captions = value['anns']
            response = model.infer({'text': coco_prompt, 'img': value['img']})
            ret = evaluator.evaluate(response, gt_captions)

            csv_line = {"index": index, "text": coco_prompt, "img": value['img'], 
                        "answer": response, "label": gt_captions, "bleu": ret["bleu"], 
                        "cider": ret["cider"], "sbert_similarity": ret["sbert_similarity"]}
            uf.handle_output(args, csv_line)
    
    elif args.task == 'mmlu':
        data_dir = '/mnt/kaichen/data/mmlu_data'
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(data_dir, "test")) if "_test.csv" in f])

        total_qestions = 0
        sub_num = {}
        for sub in subjects:
            test_df = pd.read_csv(os.path.join(data_dir, "test", sub + "_test.csv"), header=None)
            total_qestions += test_df.shape[0]
            sub_num[sub] = test_df.shape[0]

        if args.vqa_start_point == 0:
            print(f'MMLU bench has total {total_qestions} questions, num of each subject is:')
            for index, (key, value) in enumerate(sub_num.items()):
                print(f'subject {index}: {key} - {value}')
            print("")

        if args.vqa_start_point != 0:
            args.start_sub, args.start_ind = uf.mmlu_get_next_sample(args.mmlu_resume_args, sub_num)
        else:
            args.start_sub, args.start_ind = subjects[0], 0

        args.mmlu_qa_index = args.vqa_start_point
        start_flag = False
        args.ntrain = 5
        for subject in subjects:
            if subject != args.start_sub and not start_flag:
                continue
            elif subject == args.start_sub:
                start_flag = True

            dev_df = pd.read_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
            test_df = pd.read_csv(os.path.join(data_dir, "test", subject + "_test.csv"), header=None)
            uf.eval_mmlu(model, subject, dev_df, test_df, args)

            if args.mmlu_qa_index >= args.vqa_num:
                break

if __name__ == "__main__":
    main()