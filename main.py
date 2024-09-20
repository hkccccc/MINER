"""main python file"""
import os
import json
import argparse
import sys
import re
import pickle
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALL_MODALITIES = ["text", "special", "special_text", "image", "video", "audio", "prompt"]
MODILITY_SPLIT_TYPES = ["prompt", "special_text_as_one", "special_text_separate"]
MLLM_MODALITIES = { # subset of ALL_MODALITIES
    'qwen2_vl': {
        'text_vqa': ["text", "special", "special_text", "image", "prompt"],
        'coco_caption': ["text", "special", "special_text", "image", "prompt"],
        'mmlu': ["text", "special", "prompt"]
    }
}
SELECT_STRATEGIES = ["uniform", "adaptive", "LA_MU", "LU_MA", "random"]
ALL_DATASETS = ["test", "text_vqa", "coco_caption", "mmlu"]

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(description="A simple example script to demonstrate argparse.")    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "one_llm"], help="the mllm to be analyzed")
    parser.add_argument('--dataset', type=str, default='text_vqa', choices=ALL_DATASETS)
    parser.add_argument('--gpu', type=str, default='0', help="gpu_id")
    parser.add_argument('--demo', type=int, default=0, help="switch to 1 if wanna run demo with demo.sh")

    parser.add_argument('--mode', type=float, choices=[0,1,1.5,2,3], default=2)
    # mode 0: infer normally
    # mode 1: generate Impotance Score Matrix (ISM), stage1
    # mode 1.5: aggregate ISM from different datasets
    # mode 2: generate mask, stage2
    # mode 3: apply mask, stage3

    # stage1: generate ISM
    parser.add_argument('--sample_num', type=int, default=10, help="number of samples, -1 means all samples")
    # stage1.5: aggregate ISM from different datasets
    parser.add_argument('--load_ISM_sample_num', type=int, nargs='+', default=[-1,10,-1], help="ISM_x.npy for [text_vqa,coco_caption,mmlu], 0 means not use, -1 means ISM.npy")
    # stage2: generate mask
    parser.add_argument('--sum_ISM_path', type=str, default="text_vqa30_coco_caption20_mmlu19")
    parser.add_argument('--select_ratio', type=float, default=0.01, help="the ratio of selected neurons")
    parser.add_argument('--importance_metric_weights', type=float, nargs='+', default=[0,0.5,0.5,0,0], help="weights of different importance metric: [prob,mean,max,attn_k,attn_q], sum=1")
    parser.add_argument('--select_strategy', type=str, choices=SELECT_STRATEGIES, default="LU_MA")
    # stage3: apply mask
    parser.add_argument('--modality_split_type', type=str, choices=MODILITY_SPLIT_TYPES, default="special_text_as_one")
    parser.add_argument('--mask_modalities', type=str, nargs='+', default=["image"], help="choose modalities want to mask, should be subset of MLLM_MODALITIES[args.dataset]")
    parser.add_argument('--deactivation_val', type=float, default=0, help="output value of a deactivated neuron, -1 means output.min()")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from utils import func as uf
    from models import Qwen2_VL

    uf.set_seed(2024)
    if args.sample_num == -1:
        args.sample_num = {'test': 1, 'text_vqa': 5000, 'coco_caption': 5000, 'mmlu': 14042}.get(args.dataset)
    
    args_dict = vars(args)
    cmd = [sys.executable] + sys.argv[:1]
    cmd += [f"--{key} {' '.join(map(str, value)) if isinstance(value, list) else value}" for key, value in args_dict.items() if value is not None]
    cmd_str = ' '.join(cmd)
    args.vqa_start_point = 0
    args.save_score_steps = [10, 100, 500, 1000, 2500, 5000, 10000]
    args.all_modalities = ALL_MODALITIES
    args.importance_metric_types = ['prob', 'mean', 'max', 'attn_k', 'attn_q']
    
    args.demo_str = 'demo_' if args.demo else ''
    mode3_str = '/mask_csv' if args.mode == 3 else ''
    args.folder_path = f"{args.demo_str}outputs/{args.mllm}/{args.dataset}{mode3_str}"
    csv_name = 'origin'
    if args.mode == 3:
        csv_name = f"{args.sum_ISM_path}_{args.modality_split_type}_r{args.select_ratio}_{'_'.join(map(str, args.importance_metric_weights))}_{args.select_strategy}_{'_'.join(args.mask_modalities)}_deact_val{args.deactivation_val}"
    
    if args.mode == 1.5:
        ISM_list = []
        sum_ISM_path = f"{args.demo_str}outputs/{args.mllm}/"
        for dataset, ISM_sample_num in zip(ALL_DATASETS[1:], args.load_ISM_sample_num):
            if ISM_sample_num == -1:
                ISM_path = "ISM.npy"
            elif ISM_sample_num == 0:
                continue
            else:
                ISM_path = f'ISM_{ISM_sample_num}.npy'
            ISM_path = f'{args.folder_path.replace(args.dataset, dataset)}/ISM/{ISM_path}'
            if os.path.exists(ISM_path):
                with open(ISM_path, "rb") as f:
                    sample_num, ISM = pickle.load(f)
                    ISM /= sample_num # normalize to [0,1]
                    assert ISM.max() <= 1
                    ISM_list.append(ISM)
                    sum_ISM_path += f'{dataset}{sample_num}_'
                    print(f"load ISM from {ISM_path} with {sample_num} samples.")
        if len(ISM_list) == 0:
            print("Oops! ISM file not exists, load nothing!")
            sys.exit(0)
        else:
            sum_ISM = sum(ISM_list)
            occurrence_times = {}
            for ind, modality in enumerate(ALL_MODALITIES):
                count = sum(ISM[:, ind].sum() != 0 for ISM in ISM_list)
                occurrence_times[modality] = int(count)
                if count != 0:
                    sum_ISM[:, ind] /= count # normalize to [0,1]
            print(f'occurrence times of different modalities: {occurrence_times}')
            
            # create folder for sum_ISM and 3 kinds splits
            new_path = sum_ISM_path[:-1]
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            if not os.path.exists(f'{new_path}/sum_ISM.npy'):
                with open(f'{new_path}/sum_ISM.npy', "wb") as f:
                    pickle.dump(sum_ISM, f)
            else:
                print(f"{new_path}/sum_ISM.npy already exists!")
            
            # 3 different splits: prompt.npy, special_text_separate.npy, special_text_as_one.npy
            # delete modalities never occur and have conflit with current split
            with open(f"{new_path}/prompt.npy", "wb") as f:
                pickle.dump((["prompt"], sum_ISM[:, -1]), f)
            special_text_separate_list = []
            special_text_as_one_list = []
            for ind, modality in enumerate(ALL_MODALITIES):
                if sum_ISM[:, ind].sum() == 0:
                    continue
                if modality not in ["prompt", "special_text"]:
                    special_text_separate_list.append((ind, modality))
                if modality not in ["prompt", "special", "text"]:
                    special_text_as_one_list.append((ind, modality))
            with open(f"{new_path}/special_text_separate.npy", "wb") as f:
                list1, list2 = zip(*special_text_separate_list)
                pickle.dump((list2, sum_ISM[:, list1]), f)
            with open(f"{new_path}/special_text_as_one.npy", "wb") as f:
                list1, list2 = zip(*special_text_as_one_list)
                pickle.dump((list2, sum_ISM[:, list1]), f)
            sys.exit(0)
    
    if args.mode == 2:
        # find neuron masks of all possible combination of mode2 hyperparameters
        # save in 'datasetxx_xx' folder, can be updated incrementally
        args.sum_ISM_path = f"{args.demo_str}outputs/{args.mllm}/" + args.sum_ISM_path
        if os.path.exists(f"{args.sum_ISM_path}/masks.npy"):
            with open(f"{args.sum_ISM_path}/masks.npy", "rb") as f:
                mask_dict = pickle.load(f)
        else:
            mask_dict = {}
            
        def generate_mask(name_str):
            mask_index = (name_str, args.select_ratio, "_".join(map(str, args.importance_metric_weights)), args.select_strategy)
            if mask_dict.get(mask_index) is not None:
                print(f"mask of {mask_index} already exists!")
                return
            with open(f"{args.sum_ISM_path}/{name_str}.npy", "rb") as f:
                ISM_temp = pickle.load(f)
            mask_dict[mask_index] = uf.select_modality_neurons_from_importance_scores(ISM_temp, args)
            print(f"successfully save mask of {mask_index}!")
        
        for modality_split_type in MODILITY_SPLIT_TYPES:
            generate_mask(modality_split_type)
            
        with open(f"{args.sum_ISM_path}/masks.npy", "wb") as f:
            pickle.dump(mask_dict, f)
        sys.exit(0)
        
    if args.mode == 1 or args.mode == 3:
        if not os.path.exists(args.folder_path):
            os.makedirs(args.folder_path)
            if args.mode == 1:
                with open(args.folder_path + '/run.sh', mode='w', encoding='utf-8') as f:
                    f.write(cmd_str)
                sys.stdout = open(args.folder_path + '/out.txt', 'a', encoding='utf-8')
            uf.initialize_csv(f'/{csv_name}.csv', args)
        
        elif args.mode == 3 and not os.path.exists(args.folder_path + f'/{csv_name}.csv'): # other deactivation_val
            uf.initialize_csv(f'/{csv_name}.csv', args)
        
        else:
            # check whether need to resume
            if args.mode == 1:
                sys.stdout = open(args.folder_path + '/out.txt', 'a', encoding='utf-8')
            df = pd.read_csv(args.folder_path + f'/{csv_name}.csv')

            if df['index'].iloc[-1] != args.sample_num - 1:
                args.vqa_start_point = df['index'].iloc[-1] + 1
                args.mmlu_resume_args = (df["dataset name"].iloc[-1], df["sub-index"].iloc[-1])
                uf.initialize_csv(f'/{csv_name}.csv', args)
            else:
                print("Condition met, exiting the program...")
                sys.exit(0)

    # return output
    if args.mode == 3:
        args.sum_ISM_path = f"{args.demo_str}outputs/{args.mllm}/" + args.sum_ISM_path
        with open(f"{args.sum_ISM_path}/masks.npy", "rb") as f:
            mask_dict = pickle.load(f)
        
        mask_index = (args.modality_split_type, args.select_ratio, "_".join(map(str, args.importance_metric_weights)), args.select_strategy)
        args.k_inds = mask_dict[mask_index]
    
    evaluator = uf.TextEvaluation()

    # create mllm model
    if args.mllm == 'qwen2_vl':
        model = Qwen2_VL(args)
    else:
        pass # add more models

    # processing questions from various benchmarks
    if args.dataset == 'test':
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

    elif args.dataset == 'text_vqa':
        text_vqa_path = "/mnt/kaichen/data/TextVQA"
        with open(f"{text_vqa_path}/val/TextVQA_0.5.1_val.json", "r", encoding='utf-8') as f:
            text_vqa = json.load(f)['data']

        for i in range(args.vqa_start_point, args.sample_num):
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

    elif args.dataset == 'coco_caption':
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
            if index >= args.sample_num:
                break
            gt_captions = value['anns']
            response = model.infer({'text': coco_prompt, 'img': value['img']})
            ret = evaluator.evaluate(response, gt_captions)

            csv_line = {"index": index, "text": coco_prompt, "img": value['img'], 
                        "answer": response, "label": gt_captions, "bleu": ret["bleu"], 
                        "cider": ret["cider"], "sbert_similarity": ret["sbert_similarity"]}
            uf.handle_output(args, csv_line)
    
    elif args.dataset == 'mmlu':
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

            if args.mmlu_qa_index >= args.sample_num:
                break

if __name__ == "__main__":
    main()