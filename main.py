"""main python file"""
import os
import json
import argparse
import sys
import re
import pickle
import itertools
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALL_MODALITIES = ["text", "special", "special_text", "image", "video", "audio", "prompt"]
ALL_MODILITY_SPLIT_TYPES = { # store modality and index
    "prompt": {6: "prompt"},
    "special_text_as_one": {2: "special_text", 3: "image", 4: "video", 5: "audio"},
    "special_text_separate": {0: "text", 1: "special", 3: "image", 4: "video", 5: "audio"},
}
MLLM_MODALITIES = { # subset of ALL_MODALITIES
    'qwen2_vl': {
        'text_vqa': ["text", "special", "special_text", "image", "prompt"],
        'coco_caption': ["text", "special", "special_text", "image", "prompt"],
        'mmlu': ["text", "special", "prompt"],
        'msvd_qa': ["text", "special", "special_text", "video", "prompt"],
    },
    'qwen2_audio': {
        'mmlu': ["text", "special", "prompt"],
        'libri': ["text", "special", "special_text", "audio", "prompt"],
        'vocal_sound': ["text", "special", "special_text", "audio", "prompt"]
    }
}
ALL_SELECT_STRATEGIES = ["uniform", "adaptive", "LA_MU", "LU_MA", "random"]
ALL_DATASETS = ["test", "text_vqa", "coco_caption", "mmlu", "msvd_qa", "libri", "vocal_sound"]
ALL_SELECT_RATIO = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1]
ALL_IMPORTANCE_METRIC_TYPES = ['prob', 'mean', 'max', 'attn_k', 'attn_q']
ALL_SAVE_SAMPLE_NUMS = [10, 100, 500, 1000, 2500, 5000, 10000]
ALL_SAMPLE_NUMS = {
    'test': 1,
    'text_vqa': 5000,
    'coco_caption': 5000,
    'mmlu': 14042,
    'msvd_qa': 13157,
    'libri': 2620,
    'vocal_sound': 20977,
}
ALL_IMPORTANCE_METRIC_WEIGHTS = [
    [1,0,0,0,0], # prob
    [0,1,0,0,0], # mean
    [0,0,1,0,0], # max
    [0,0,0,1,0], # attn_k
    [0,0,0,0,1], # attn_q
    [0,0.5,0.5,0,0],
    [0,0,0,0.5,0.5],
    [0,0.25,0.25,0.5,0],
    [0,0.25,0.25,0,0.5],
    [0.2,0.2,0.2,0.2,0.2],
]

def main():
    """
    main function
    """
    parser = argparse.ArgumentParser(description="A simple example script to demonstrate argparse.")    
    parser.add_argument('--mllm', type=str, default='qwen2_vl', choices=["qwen2_vl", "qwen2_audio"], help="the mllm to be analyzed")
    parser.add_argument('--dataset', type=str, default='text_vqa', choices=ALL_DATASETS)
    parser.add_argument('--gpu', type=str, default='0', help="gpu_id")
    parser.add_argument('--demo', type=int, default=0, help="switch to 1 if wanna run demo with demo.sh")

    parser.add_argument('--mode', type=float, choices=[0,1,1.5,2,3], default=2)
    # mode 0: infer normally
    # mode 1: generate Impotance Score Matrix (ISM), stage1
    # mode 1.5: aggregate ISM from different datasets
    # mode 2: generate mask, stage2 (store all possible masks in advance, time consuming)
    # mode 3: apply mask, stage3 (mode 2 can be skipped, mode 3 contains the mask generation process)

    # stage1: generate ISM
    parser.add_argument('--sample_num', type=int, default=-1, help="number of samples, -1 means all samples")
    parser.add_argument('--sample_start_ind', type=int, default=0)
    # stage1.5: aggregate ISM from different datasets
    parser.add_argument('--load_ISM_sample_num', type=int, nargs='+', default=[-1,-1,-1,-1,-1,-1], help="ISM_x.npy for [text_vqa,coco_caption,mmlu,msvd_qa,libri,vocal_sound], 0 means not use, -1 means ISM.npy")
    # stage2: generate mask
    parser.add_argument('--sum_ISM_path', type=str, default="text_vqa5000_coco_caption5000_mmlu14042")
    parser.add_argument('--select_ratio', type=float, default=0.02, choices=ALL_SELECT_RATIO, help="the ratio of selected neurons")
    parser.add_argument('--importance_metric_weights', default=[0,0.25,0.25,0.5,0], type=float, nargs='+', help="weights of different importance metric: [prob,mean,max,attn_k,attn_q], sum=1")
    parser.add_argument('--select_strategy', default="LA_MU", type=str, choices=ALL_SELECT_STRATEGIES)
    # stage3: apply mask
    parser.add_argument('--modality_split_type', default="special_text_separate", type=str, choices=list(ALL_MODILITY_SPLIT_TYPES.keys()))
    parser.add_argument('--mask_modalities', type=str, nargs='+', default=["all_modalities"], help="choose modalities want to mask, should be subset of MLLM_MODALITIES[args.dataset]")
    parser.add_argument('--deactivation_val', type=float, default=0, help="output value of a deactivated neuron, -1 means output.min()")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from utils import func as uf
    from utils import crop as uc
    from models import Qwen2_VL, Qwen2_Audio

    uf.set_seed(2024)
    args.sample_num = ALL_SAMPLE_NUMS.get(args.dataset, args.sample_num) if args.sample_num == -1 else args.sample_num
    args.sample_num_start_from = 0
    
    args.all_save_sample_nums = ALL_SAVE_SAMPLE_NUMS
    args.all_modalities = ALL_MODALITIES
    args.all_importance_metric_types = ALL_IMPORTANCE_METRIC_TYPES
    
    # range(args.sample_start_ind, sargs.ample_end_ind)，左闭右开
    if args.sample_start_ind + args.sample_num < ALL_SAMPLE_NUMS[args.dataset]:
        args.sample_end_ind = args.sample_start_ind + args.sample_num
    else:
        args.sample_end_ind = ALL_SAMPLE_NUMS[args.dataset]
    args.sample_str = f'start{args.sample_start_ind}_end{args.sample_end_ind}'

    args.demo_prefix = 'demo_' if args.demo else ''
    args.mllm_path = f"{args.demo_prefix}outputs/{args.mllm}"
    args.mllm_dataset_path = f"{args.mllm_path}/{args.dataset}"
    args.mllm_dataset_ISM_path = f"{args.mllm_dataset_path}/ISM"
    
    # create folder and initialize csv file
    if args.mode in [1, 3]:
        base_folder_path = args.mllm_dataset_path if args.mode == 1 else f"{args.mllm_dataset_path}/mask_csv"
        args.csv_name = "origin" if args.mode == 1 else f"{args.sum_ISM_path}--{args.modality_split_type}--{args.select_ratio}--{'_'.join(map(str, args.importance_metric_weights))}--{args.select_strategy}--{'_'.join(args.mask_modalities)}--{args.deactivation_val}"
        args.csv_path = f"{base_folder_path}/{args.csv_name}_{args.sample_str}.csv"

        if not os.path.exists(base_folder_path):
            os.makedirs(base_folder_path)
            uf.initialize_csv(f'/{args.csv_name}.csv', args)
            
        elif args.mode == 3 or not os.path.exists(args.csv_path):
            uf.initialize_csv(f'/{args.csv_name}.csv', args)        
            
        else: # check whether need to resume
            df = pd.read_csv(args.csv_path)
            if df['index'].iloc[-1] != args.sample_num - 1:
                args.sample_num_start_from = df['index'].iloc[-1] + 1
                args.mmlu_resume_args = (df["dataset name"].iloc[-1], df["sub-index"].iloc[-1])
                uf.initialize_csv(f'/{args.csv_name}.csv', args)
            else:
                print("Condition met, exiting the program...")
                sys.exit(0)
    
    if args.mode == 1.5:
        ISM_list = []
        sum_ISM_path = f"{args.mllm_path}/sum_ISM/"

        assert len(args.load_ISM_sample_num) == len(ALL_DATASETS) - 1
        for dataset, ISM_sample_num in zip(ALL_DATASETS[1:], args.load_ISM_sample_num):
            if ISM_sample_num == 0: continue
            ISM_file = "ISM.npy" if ISM_sample_num == -1 else f'ISM_{ISM_sample_num}.npy'
            ISM_path = os.path.join(args.mllm_dataset_ISM_path.replace(args.dataset, dataset), ISM_file)
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
            # Normalize based on the different frequencies of modalities in the datasets.
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
            os.makedirs(new_path, exist_ok=True)
            file_path = os.path.join(new_path, 'sum_ISM.npy')
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    pickle.dump(sum_ISM, f)
            else:
                print(f"{file_path} already exists!")

            # 3 different splits: prompt.npy, special_text_separate.npy, special_text_as_one.npy
            # delete modalities never occur and have conflit with current split
            for split_type, index_modality_map in ALL_MODILITY_SPLIT_TYPES.items():
                modality_list, ISM_index_list = [], []
                for index, modality in index_modality_map.items():
                    if sum_ISM[:, index].sum() == 0: continue
                    modality_list.append(modality)
                    ISM_index_list.append(index)
                with open(f"{new_path}/{split_type}.npy", "wb") as f:
                    pickle.dump((modality_list, sum_ISM[:, ISM_index_list]), f)
                    
            sys.exit(0)
    
    def generate_mask_of_modality_specific_neurons(
        args,
        sum_ISM_path=None,
        modality_split_type=None,
        select_ratio=None,
        importance_metric_weights=None,
        select_strategy=None,
    ):
        """
        All parameters are optional. If none are specified, all possible values will be iterated.
        input: (
            sum_ISM_path: text_vqa5000_coco_caption5000_mmlu14042,
            modality_split_type,
            select_ratio,
            importance_metric_weights (list of length 5),
            select_strategy,
        )
        output: (
            sum_ISM_path/masks.npy is updated,
            last modality_specific_neuron_mask is saved in args.modality_specific_neuron_mask
        )
        """
        all_sum_ISM_paths = []
        for subfolder in Path(f"{args.mllm_path}/sum_ISM").rglob('*'):
            if subfolder.is_dir() and re.search(r'(' + '|'.join(ALL_DATASETS) + r')\d+', str(subfolder)):
                if sum_ISM_path is not None and subfolder.name != sum_ISM_path:
                    continue
                all_sum_ISM_paths.append(subfolder.name)
        
        for temp_sum_ISM_path in all_sum_ISM_paths:
            sum_ISM_path = f"{args.mllm_path}/sum_ISM/{temp_sum_ISM_path}"
            masks_path = f"{sum_ISM_path}/masks.npy"
            if os.path.exists(masks_path): # update incrementally
                with open(masks_path, "rb") as f:
                    mask_dict = pickle.load(f)
            else:
                mask_dict = {}
            
            # iterate all possible combination of parameters
            cartesian_product = list(itertools.product(
                list(ALL_MODILITY_SPLIT_TYPES.keys()),
                ALL_SELECT_RATIO,
                ALL_IMPORTANCE_METRIC_WEIGHTS,
                ALL_SELECT_STRATEGIES,
            ))
            for item in cartesian_product:
                if (modality_split_type is not None and item[0] != modality_split_type) or \
                   (select_ratio is not None and item[1] != select_ratio) or \
                   (importance_metric_weights is not None and item[2] != importance_metric_weights) or \
                   (select_strategy is not None and item[3] != select_strategy):
                    continue
                
                # These parameters can uniquely determine a mask.
                mask_index = (item[0], item[1], "_".join(map(str, item[2])), item[3])
                
                if mask_dict.get(mask_index) is not None:
                    args.modality_specific_neurons = mask_dict.get(mask_index)[0]
                    print(f"mask of {temp_sum_ISM_path}, {mask_index} already exists!")
                    continue

                with open(f"{sum_ISM_path}/{item[0]}.npy", "rb") as f:
                    ISM_temp = pickle.load(f)
                ret = uf.select_modality_neurons_from_importance_scores(ISM_temp, mask_index)
                mask_dict[mask_index] = ret # (mask, df)
                args.modality_specific_neurons = ret[0]
                print(f"successfully save mask of {temp_sum_ISM_path}, {mask_index}!")
        
            with open(masks_path, "wb") as f:
                pickle.dump(mask_dict, f)

        if not hasattr(args, 'modality_specific_neurons'):
            print("Oops! something wrong, fail to generate mask!")
            inputs = [modality_split_type, importance_metric_weights, select_ratio, select_strategy]
            choices = [list(ALL_MODILITY_SPLIT_TYPES.keys()), ALL_IMPORTANCE_METRIC_WEIGHTS, ALL_SELECT_RATIO, ALL_SELECT_STRATEGIES]
            print(uf.check_values_in_lists(inputs, choices))
            sys.exit(0)
        
    if args.mode == 2:
        # save all possible parameter combinations
        generate_mask_of_modality_specific_neurons(
            args,
        )
        sys.exit(0)

    if args.mode == 3:
        generate_mask_of_modality_specific_neurons(
            args,
            sum_ISM_path=args.sum_ISM_path,
            modality_split_type=args.modality_split_type,
            select_ratio=args.select_ratio,
            importance_metric_weights=args.importance_metric_weights,
            select_strategy=args.select_strategy,
        )        
    
    evaluator = uf.TextEvaluation()

    # create mllm model
    if args.mllm == 'qwen2_vl':
        model = Qwen2_VL(args)
    elif args.mllm == 'qwen2_audio':
        model = Qwen2_Audio(args)
    else:
        pass # add more models
    
    # processing questions from various benchmarks
    if args.dataset == 'test':
        if args.mllm == 'qwen2_vl':
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
            print(model.infer(data1))
        elif args.mllm == 'qwen2_audio':
            data = {
                'text': "What's that sound?",
                'audio': "scripts/glass-breaking.mp3"
            }
            print(model.infer(data))

    elif args.dataset == 'text_vqa':
        text_vqa_path = "/mnt/kaichen/data/TextVQA"
        with open(f"{text_vqa_path}/val/TextVQA_0.5.1_val.json", "r", encoding='utf-8') as f:
            text_vqa = json.load(f)['data']

        for index in range(args.sample_start_ind, args.sample_end_ind):
            vqa = text_vqa[index]
            data = {
                'text': vqa['question'],
                'img': f"{text_vqa_path}/val/train_images/{vqa['image_id']}.jpg"
            }
            response = model.infer(data)
            cor = uf.check_acc(response, vqa['answers'])
            csv_line = {"index": index, "text": data["text"], "img": data["img"], 
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
        
        for index, (_, value) in enumerate(coco_data.items()):
            if args.sample_start_ind <= index < args.sample_end_ind:
                gt_captions = value['anns']
                response = model.infer({'text': coco_prompt, 'img': value['img']})
                ret = evaluator.evaluate(response, gt_captions)

                csv_line = {"index": index, "text": coco_prompt, "img": value['img'], 
                            "answer": response, "label": gt_captions, "bleu": ret["bleu"], 
                            "cider": ret["cider"], "sbert_similarity": ret["sbert_similarity"]}
                uf.handle_output(args, csv_line)
    
    elif args.dataset == 'mmlu':
        subjects = sorted(f[:-9] for f in os.listdir('/mnt/kaichen/data/mmlu_data/test') if f.endswith('_test.csv'))
        current_subject = None
        current_subject_count = 0
        choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        mmlu_data = pd.read_parquet("/mnt/kaichen/data/mmlu_data/mmlu-test-all.parquet", engine='pyarrow')

        dev_test_dict = {}
        for subject in subjects:
            dev_df = pd.read_csv(f"/mnt/kaichen/data/mmlu_data/dev/{subject}_dev.csv", header=None)[:5]
            test_df = pd.read_csv(f"/mnt/kaichen/data/mmlu_data/test/{subject}_test.csv", header=None)
            dev_test_dict[subject] = {'dev': dev_df, 'test': test_df}

        for index, case in mmlu_data.iterrows():
            subject = case['subject']
            if subject != current_subject:
                current_subject = subject
                current_subject_count = 0
            else:
                current_subject_count += 1
                
            if args.sample_start_ind <= index < args.sample_end_ind:
                k = 5
                prompt_end = uf.format_example(dev_test_dict[subject]['test'], current_subject_count, include_answer=False)
                train_prompt = uf.gen_prompt(dev_test_dict[subject]['dev'], subject, k)
                prompt = train_prompt + prompt_end
                
                while uc.crop(prompt) != prompt:
                    k -= 1
                    train_prompt = uf.gen_prompt(dev_test_dict[subject]['dev'], subject, k)
                    prompt = train_prompt + prompt_end

                pred = model.infer({'text': prompt})
                label = choices[case['answer']]
                csv_line = {"index": index, "dataset name": subject, "sub-index": current_subject_count, "text": prompt,
                            "answer": pred, "label": label, "correct": (pred[0] == label)}
                uf.handle_output(args, csv_line)
    
    elif args.dataset=="msvd_qa":
        dataset = uf.MSVD()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        index = 0
        for data in dataloader:
            if args.sample_start_ind <= index < args.sample_end_ind:
                image_paths, questions, question_ids, answers = data
                response = model.infer({'text': questions[0], 'video': image_paths[0]})
                cor = uf.check_acc(response, answers)
                csv_line = {"index": index, "text": questions[0], "video": image_paths[0], 
                            "answer": response, "label": answers, "correct": cor}
                uf.handle_output(args, csv_line)
            index += 1
    
    elif args.dataset == 'libri':
        # Word Recognition Rate(WRR) = 1 - WER
        paths, labels = uf.load_libri()
        prompt = "Please transcribe the following audio directly into plain text without any additional explanations, prefixes, or descriptions. Only output the transcription of the spoken content in the audio."

        for index, audio_path in enumerate(paths):
            if args.sample_start_ind <= index < args.sample_end_ind:
                label = labels[index]
                response = model.infer({"text": prompt, "audio": audio_path})
                csv_line = {"index": index, "text": prompt, "audio": audio_path, 
                            "answer": response, "label": label, "wrr": 1 - uf.wer(label, response)}
                uf.handle_output(args, csv_line)
            
    elif args.dataset == 'vocal_sound':
        label_dict = {
            '/m/01j3sz': 'Laughter',
            '/m/07plz5l': 'Sigh',
            '/m/01b_21': 'Cough',
            '/m/0dl9sf8': 'Throat clearing',
            '/m/01hsr_': 'Sneeze',
            '/m/07ppn3j': 'Sniff'
        }
        prompt = """You are a sound classification model. Your task is to classify a given audio sample into one of the following categories based on its content:
                 1. Laughter
                 2. Sigh
                 3. Cough
                 4. Throat clearing
                 5. Sneeze
                 6. Sniff
                 Please analyze the audio sample and provide the corresponding category name.
                 """
        
        json_path = "/mnt/kaichen/data/vocal_sound_release_16k/datafiles/all.json"
        old_path_str = "/data/sls/scratch/yuangong/vocalsound2/data/vs_processed/data_16k/"
        new_path_str = "/mnt/kaichen/data/vocal_sound_release_16k/audio_16k/"

        with open(json_path, 'r') as f:
            file_lst = json.load(f)['data']
        
        for index, item in enumerate(file_lst):
            if args.sample_start_ind <= index < args.sample_end_ind:
                audio_path, label = item['wav'].replace(old_path_str, new_path_str), item['labels']
                response = model.infer({"text": prompt, "audio": audio_path})
                csv_line = {"index": index, "text": prompt, "audio": audio_path, 
                            "answer": response, "label": label_dict[label], "correct": (label_dict[label] in response)}
                uf.handle_output(args, csv_line)

if __name__ == "__main__":
    main()