"""utils functions to assist other python files"""
import random
import os
import csv
import torch
import pickle
import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from pathlib import Path
from itertools import combinations
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader

from .crop import crop

class MSVD(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.datas = json.load(open('/mnt/kaichen/data/MSVD/test_qa.json'))
        map_ids =[x.strip().split(' ') for x in open('/mnt/kaichen/data/MSVD/youtube_mapping.txt').readlines()]
        self.id_to_video_ids = {x[1]:x[0] for x in map_ids}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id = 'vid'+str(data['video_id'])
        video_name = self.id_to_video_ids[video_id] + '.avi'
        image_path = os.path.join("/mnt/kaichen/data/MSVD/YouTubeClips", video_name)
        question_id = data['id']
        question = data['question'] + '\nAnswer the question using a single word or phrase.'
        answer = data['answer']
        return image_path, question, question_id, answer

class TextEvaluation:
    """
    evaluation between predicted sentence and ground truth sentences
    """
    def __init__(self):
        self.sbert_model = SentenceTransformer('/mnt/kaichen/data/paraphrase-MiniLM-L6-v2')

    def calculate_bleu(self, input_sentence, reference_sentences):
        """
        calculate BLEU score
        """
        input_tokens = input_sentence.split()
        reference_tokens_list = [ref.split() for ref in reference_sentences]
        smoothing_function = SmoothingFunction().method1

        bleu_scores = [
            sentence_bleu([ref], input_tokens, smoothing_function=smoothing_function) 
            for ref in reference_tokens_list
        ]
        result = {
            'max': max(bleu_scores),
            'min': min(bleu_scores),
            'mean': np.mean(bleu_scores),
            'scores': bleu_scores
        }
        return result

    def calculate_sbert_similarity(self, input_sentence, reference_sentences):
        """
        compute Sentence-BERT similarity
        """
        input_embedding = self.sbert_model.encode(input_sentence, convert_to_tensor=True)
        reference_embeddings = self.sbert_model.encode(reference_sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(input_embedding, reference_embeddings).cpu().numpy().flatten().tolist()
        result = {
            'max': max(cosine_scores),
            'min': min(cosine_scores),
            'mean': np.mean(cosine_scores),
            'scores': cosine_scores
        }
        return result

    def calculate_tfidf(self, captions):
        """
        compute TF-IDF
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(captions)
        return tfidf_matrix, vectorizer

    def calculate_cider(self, input_sentence, reference_sentences):
        """
        compute score of CIDEr
        """
        references = reference_sentences.copy()
        references.append(input_sentence)

        tfidf_matrix, _ = self.calculate_tfidf(references)

        sim_matrix = cosine_similarity(tfidf_matrix)
        cider_scores = sim_matrix[-1, :-1].tolist()
        result = {
            'max': max(cider_scores),
            'min': min(cider_scores),
            'mean': np.mean(cider_scores),
            'scores': cider_scores
        }
        return result

    def evaluate(self, input_sentence, reference_sentences):
        """
        compute all metrics
        """
        results = {
            'bleu': self.calculate_bleu(input_sentence, reference_sentences),
            'sbert_similarity': self.calculate_sbert_similarity(input_sentence, reference_sentences),
            'cider': self.calculate_cider(input_sentence, reference_sentences)
        }
        return results

def split_mask_prefix(tasks, file_name):
    """
    split the mask prefix into "task" and "sample_ratio"
    """
    for task in tasks:
        if file_name.startswith(task):
            params = file_name[len(task):]
            params = params.lstrip('_')
            return task, params if params else None
    return None, None

def append_suffix_to_filename(file_path, suffix="_cp"):
    """
    add suffix
    """
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}{suffix}{ext}"
    new_file_path = os.path.join(directory, new_filename)
    return new_file_path

def is_cp_suffix(file_path):
    """
    check the cp suffix
    """
    _, filename = os.path.split(file_path)
    name, _ = os.path.splitext(filename)
    return name.endswith('_cp')

def karpathy_coco_test():
    """
    generate karpathy coco testset, 5000 images with multiple captions in total
    """
    coco = COCO('/mnt/kaichen/data/coco/annotations_trainval2014/captions_val2014.json')
    root_img_path = '/mnt/kaichen/data/coco/val2014/'
    old_imgIds = coco.getImgIds()
    with open('/mnt/kaichen/data/coco/karparthy_split2014/coco_test.txt', 'r', encoding='utf-8') as f:
        kar_test = [line.strip() for line in f]

    imgIds = []
    for img_id in old_imgIds:
        filename = coco.loadImgs(img_id)[0]['file_name']
        if filename in kar_test:
            img_path = root_img_path + filename
            imgIds.append((img_id, img_path))

    ground_truths = {img_id: {'img': img_path, 'anns': []} for (img_id, img_path) in imgIds}

    for img_id in imgIds:
        img_id = img_id[0]
        annIds = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            ground_truths[img_id]['anns'].append(ann['caption'])
    return ground_truths

def set_seed(seed):
    """
    set seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_acc(response, ans):
    """
    Check if one of the answers appears in response
    """
    out = response.lower()
    for item in ans:
        if item.lower() in out:
            return True
    return False

def min_max_normalize(tensor):
    """
    min max normalization
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

def save_ISM_with_token_mask_in_one_layer(old_tensor, modality_info, args):
    """
    save ISM of each neuron in one layer
    """    
    index, modality, layer_index = modality_info
    mask = args.input_modality_masks.get(modality)
    if mask is None or mask.sum() == 0:
        return
    tensor = old_tensor.clone()[mask, :]
    flat_tensor = tensor.view(-1, tensor.shape[-1])

    importance_dict = {}
    # prob
    positive_counts = (tensor > 0).sum(dim=0)
    total_tokens = tensor.size(0)
    prob_val = positive_counts / total_tokens
    importance_dict['prob'] = min_max_normalize(prob_val)
    # mean
    mean_val = torch.mean(flat_tensor, dim=0)
    importance_dict['mean'] = min_max_normalize(mean_val)
    # max
    max_val, _ = torch.max(flat_tensor, dim=0)
    importance_dict['max'] = min_max_normalize(max_val)
    # attn
    mask = mask.squeeze(0)
    attn_score = args.attn_score.squeeze(0)
    attn_score = attn_score[mask, :][:, mask]
    # attn_k
    attn_k = F.softmax(attn_score, dim=0) # dim0, softmax on column / key
    attn_val_k = torch.matmul(attn_k.T, flat_tensor)
    attn_val_k = torch.sum(attn_val_k, dim=0)
    importance_dict['attn_k'] = min_max_normalize(attn_val_k)
    # attn_q
    attn_q = F.softmax(attn_score, dim=1) # dim1, softmax on row / query
    attn_val_q = torch.matmul(attn_q, flat_tensor)
    attn_val_q = torch.sum(attn_val_q, dim=0)
    importance_dict['attn_q'] = min_max_normalize(attn_val_q)

    # save ISM of layer neurons to modality tokens
    for ind, metric_type in enumerate(args.all_importance_metric_types):
        args.ISM_of_one_sample[ind, index, layer_index] = importance_dict[metric_type]

def find_topK_values(data_dict, ret, K, return_flag=False): # no overlap
    if return_flag:
        ret = {'-': []}
    combined_tensor = sum(data_dict.values())
    _, topk_indices = torch.topk(combined_tensor.flatten(), K)
    
    indices_2d = [torch.div(index, combined_tensor.shape[1], rounding_mode='floor').item() for index in topk_indices], \
                 [index % combined_tensor.shape[1] for index in topk_indices]
        
    for i in range(K):
        row, col = indices_2d[0][i], indices_2d[1][i]
        max_tensor_name = None
        max_value = float('-inf')
        
        for name, tensor in data_dict.items():
            if tensor[row, col] > max_value:
                max_value = tensor[row, col]
                max_tensor_name = name
        ret[max_tensor_name].append((int(row), int(col)))
    return ret if return_flag else None

def analyse_modality_specific_neurons(input_dict, L):
    modalities = list(input_dict.keys())
    modality_combinations = []
    for i in range(1, len(modalities) + 1):
        modality_combinations.extend(combinations(modalities, i))

    columns = ['-'.join(combo) for combo in modality_combinations]
    df = pd.DataFrame(0, index=list(range(L)) + ['total'], columns=columns)

    for modality, neurons in input_dict.items():
        for layer in range(L):
            neuron_count = sum(1 for l, _ in neurons if l == layer)
            df.at[layer, modality] = neuron_count

    for combo in modality_combinations:
        combo_name = '-'.join(combo)
        if len(combo) > 1:
            for layer in range(L):
                neurons_per_layer = [set(n for l, n in input_dict[modality] if l == layer) for modality in combo]
                intersection = set.intersection(*neurons_per_layer) if neurons_per_layer else set()
                df.at[layer, combo_name] = len(intersection)
    df.loc['total'] = df.sum(axis=0)
    return df

def select_modality_neurons_from_importance_scores(sum_ISM, mask_index):
    """
    input: ISM, select_ratio, importance_metric_weights, select_strategy
    output: {'text': [(0, 22),...]}
    """
    _, select_ratio, importance_metric_weights, select_strategy = mask_index
    importance_metric_weights = [float(i) for i in importance_metric_weights.split('_')]
    
    modalities, ISM = sum_ISM
    importance_metric_weights = torch.tensor(importance_metric_weights).view((5,1,1,1)).to(ISM.device)
    weighted_ISM = (ISM * importance_metric_weights).sum(dim=0)
    
    weighted_ISM_dict = {}
    M, L, N = len(modalities), ISM.shape[-2], ISM.shape[-1]
    K = int(select_ratio * L * N) # 5304
    
    modality_specific_neurons = {modality: [] for modality in (modalities + ["random"])}
    for index, modality in enumerate(modalities):
        weighted_ISM_dict[modality] = weighted_ISM[index]
        
    if select_strategy == "random":
        modality_specific_neurons["random"] = [(random.randint(0, L - 1), random.randint(0, N - 1)) for _ in range(K)]
    elif select_strategy == "adaptive":
        find_topK_values(weighted_ISM_dict, modality_specific_neurons, K)
    elif select_strategy == "LA_MU":
        for modal, modal_scores in weighted_ISM_dict.items():
            find_topK_values({modal: modal_scores}, modality_specific_neurons, K // M)           
    elif select_strategy == "LU_MA":
        Map = {i: modalities[i] for i in range(M)}
        for layer in range(L):
            sub_dict = {'-': torch.stack([v[layer] for v in weighted_ISM_dict.values()])}
            ret = find_topK_values(sub_dict, modality_specific_neurons, K // L, return_flag=True)['-']
            for (modal_ind, neuron_ind) in ret:
                modality_specific_neurons[Map[modal_ind]].append((layer, neuron_ind))            
    elif select_strategy == "uniform":
        for layer in range(L):
            matrix = torch.zeros(L, N).to(ISM.device)
            for modal, modal_scores in weighted_ISM_dict.items():
                matrix[layer] = modal_scores[layer]
                find_topK_values({modal: matrix}, modality_specific_neurons, K // (M * L))
    
    modality_specific_neurons = {k: v for k, v in modality_specific_neurons.items() if v}
    analyse_df = analyse_modality_specific_neurons(modality_specific_neurons, L)
    return modality_specific_neurons, analyse_df

def create_activation_hook(layer_index, args):
    """
    hook for activation of mlp neurons in llm
    """
    def activation_hook(module, input, output):
        if args.mode == 1 and output.shape[:-1] == args.prompt_mask_shape:
            for index, modality in enumerate(args.all_modalities):
                save_ISM_with_token_mask_in_one_layer(output, (index, modality, layer_index), args)
        
        elif args.mode == 3:
            if args.mask_modalities[0] == "all_modalities":
                wanna_mask_modalities = list(args.modality_specific_neurons.keys())
            else:
                supported_modalities = list(args.modality_specific_neurons.keys())
                wanna_mask_modalities = args.mask_modalities
                if not set(wanna_mask_modalities).issubset(set(supported_modalities)):
                    print(f"Unsupported mask modalities: {wanna_mask_modalities}, {args.modality_split_type} only support {supported_modalities}!")
                    sys.exit(0)
            
            all_positions = []
            for modality, neurons in args.modality_specific_neurons.items():
                if modality not in wanna_mask_modalities:
                    continue
                all_positions += neurons
                
            layer_positions = [t[1] for t in all_positions if t[0] == layer_index]
            output[..., layer_positions] = output.min() if args.deactivation_val == -1 else args.deactivation_val            
        return output
    return activation_hook

def save_attn_matrix(attention_matrix, name, args):
    attention_matrix = attention_matrix.to(torch.float32).squeeze(0).cpu().numpy()
    # token_names = [i for i in range(len(attention_matrix[-1]))]
    token_names = list(args.input_ids.squeeze(0).cpu().numpy())
    plt.figure(figsize=(40, 30))

    sns.heatmap(attention_matrix, xticklabels=token_names, yticklabels=token_names, cmap='viridis', annot=False, fmt=".2f")
    plt.title("Attention Heatmap")
    plt.xlabel("Tokens (Key)")
    plt.ylabel("Tokens (Query)")
    plt.savefig(name, dpi=300, bbox_inches='tight')

def create_attention_hook(layer_index, args):
    """
    hook for attention layer in llm
    """
    def attn_hook(module, input, output):
        args.attn_score = module.attn_score
        # save_attn_matrix(args.attn_score, f'{layer_index}.png', args)
        return output
    return attn_hook

def initialize_csv(csv_name, args):
    """
    initialize csv file
    """
    args.csv_fieldnames = ["index", "dataset name", "sub-index", "text", "img", "video", "audio", "answer", "label"]
    args.csv_file = open(args.csv_path, mode='a+', newline='', encoding='utf-8')
    def initialize_csv_writer(args, add_lst):
        args.csv_fieldnames += add_lst
        args.writer = csv.DictWriter(args.csv_file, fieldnames=args.csv_fieldnames)
        args.csv_file.seek(0, 0)
        if args.csv_file.read(1) == '':
            args.csv_file.seek(0, 0)
            args.writer.writeheader()
    if args.dataset in ["text_vqa", "mmlu", "msvd_qa"]:
        initialize_csv_writer(args, ["correct"])
    elif args.dataset == "coco_caption":
        initialize_csv_writer(args, ["bleu", "sbert_similarity", "cider"])

def handle_output(args, csv_line):
    """
    decide whether to print or write in csv
    """
    if args.mode != 0:
        args.writer.writerow(csv_line)
        args.csv_file.flush()
    else:
        for key, value in csv_line.items():
            print(f'{key}: {value}')
        print("")

def format_subject(subject):
    """
    format subject
    """
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    """
    format example
    """
    choices = ["A", "B", "C", "D"]
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {df.iloc[idx, j+1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {df.iloc[idx, k + 1]}\n\n"
    return prompt

def gen_prompt(train_df, subject, k=-1):
    """
    generate prompt
    """
    prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval_mmlu(model, subject, dev_df, test_df, args, re_args=None):
    """
    answer the questions from subject one by one
    """
    k = args.ntrain
    start = args.start_ind if args.start_sub == subject else 0
    
    for i in range(start, test_df.shape[0]):
        if args.mmlu_qa_index >= args.sample_num:
            return

        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        pred = model.infer({'text': prompt})

        csv_line = {"index": args.mmlu_qa_index, "dataset name": subject, "sub-index": i, "text": prompt,
                    "answer": pred, "label": label, "correct": (pred[0] == label)}
        args.mmlu_qa_index += 1
        handle_output(args, csv_line)
    return

def mmlu_get_next_sample(current_sample, counts):
    """
    get next sample
    """
    current_category, current_index = current_sample
    # Determine if we need to move to the next category
    if current_index + 1 < counts[current_category]:
        # Move to the next index within the same category
        return (current_category, current_index + 1)
    else:
        # Move to the next category
        categories = list(counts.keys())
        current_category_index = categories.index(current_category)
        if current_category_index + 1 < len(categories):
            next_category = categories[current_category_index + 1]
            return (next_category, 0)
        else:
            # No more categories
            return None
        
def check_values_in_lists(values, lists):
    violations = []
    
    for i, (value, lst) in enumerate(zip(values, lists)):
        if value not in lst:
            violations.append(f"Value {value} at index {i} not in list {lst}")
    
    if not violations:
        return True
    else:
        return violations