"""utils functions to assist other python files"""
import random
import os
import csv
import torch
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from pathlib import Path
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .crop import crop

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

def get_neuron_importance_scores_in_layer(old_tensor, mask, args):
    """
    get sumed importance score of each neuron in one layer
    """
    tensor = old_tensor.clone()[mask, :]
    flat_tensor = tensor.view(-1, tensor.shape[-1])

    # prob
    positive_counts = (tensor > 0).sum(dim=0)
    total_tokens = tensor.size(0)
    prob_val = positive_counts / total_tokens
    # mean
    mean_val = torch.mean(flat_tensor, dim=0)
    # max
    max_val, _ = torch.max(flat_tensor, dim=0)
    # attn
    mask = mask.squeeze(0)
    attn_score = args.attn_score.squeeze(0)
    attn_score = attn_score[mask, :][:, mask]
    attn_k = F.softmax(attn_score, dim=0) # dim0, softmax on column / key
    attn_val_k = torch.matmul(attn_k.T, flat_tensor)
    attn_q = F.softmax(attn_score, dim=1) # dim1, softmax on row / query
    attn_val_q = torch.matmul(attn_q, flat_tensor)

    values = [prob_val, mean_val, max_val, torch.sum(attn_val_k, dim=0), torch.sum(attn_val_q, dim=0)]
    return {key: min_max_normalize(value) for key, value in zip(args.score_keys, values)}


def fill_tuple(template, value_tuple):
    """
    填充模板中的 -1 位置，生成新的 tuple。
    """
    template_list = list(template)
    value_iter = iter(value_tuple)
    for i, val in enumerate(template_list):
        if val == -1:
            template_list[i] = next(value_iter)
    return tuple(template_list)

def top_k_pos(matrix, template, K, K2=None):
    """
    Helper function to get the top K (modality, layer, neuron) tuples from a 3D matrix.
    """
    # Flatten the matrix and get the indices of the top K values
    flat_matrix = matrix.flatten()
    vals = None
    ret_dic = {}
    if K2 is not None:
        flat_indices = torch.topk(flat_matrix, K2).indices[K:]
        vals = torch.topk(flat_matrix, K2).values[K:]
    else:
        flat_indices = torch.topk(flat_matrix, K).indices
        vals = torch.topk(flat_matrix, K).values
    # Convert the flat indices to 3D indices (modality, layer, neuron)
    ret = list(zip(*torch.unravel_index(flat_indices, matrix.shape)))
    int_list = [fill_tuple(template, tuple(tensor.item() for tensor in tup)) for tup in ret]
    vals = vals.tolist()
    assert len(int_list) == len(vals)
    for i, val in enumerate(vals):
        ret_dic[val] = int_list[i]
    return ret_dic

def select_k_neurons(matrix, modals, K, mode):
    """
    select k neurons from importance matric [M,L,N]
    """
    M, L, N = matrix.shape
    base_dic = {}
    remain_dic = {}
    if mode == "adaptive":
        # Select top K from the entire MLN matrix as a 1D vector.
        K1 = K // 1
        remain_K = K - K1 * 1
        base_dic.update(top_k_pos(matrix, (-1, -1, -1), K))
        remain_dic.update(top_k_pos(matrix, (-1, -1, -1), K1, K1+1))

    elif mode == "layer_uniform":
        # Split along layers, select top K/L per layer.
        K1 = K // L
        remain_K = K - K1 * L
        for l in range(L):
            sub_matrix = matrix[:, l, :]
            base_dic.update(top_k_pos(sub_matrix, (-1, l, -1), K1))
            remain_dic.update(top_k_pos(sub_matrix, (-1, l, -1), K1, K1+1))

    elif mode == "modal_uniform":
        # Split along modalities, select top K/M per modality.
        K1 = K // M
        remain_K = K - K1 * M
        for m in range(M):
            sub_matrix = matrix[m, :, :]
            base_dic.update(top_k_pos(sub_matrix, (m, -1, -1), K1))
            remain_dic.update(top_k_pos(sub_matrix, (m, -1, -1), K1, K1+1))
        
    elif mode == "uniform":
        # Uniform: Split into ML parts, select top K/ML from each.
        K1 = K // (M * L)
        remain_K = K - K1 * M * L
        for m in range(M):
            for l in range(L):
                sub_matrix = matrix[m, l, :]
                base_dic.update(top_k_pos(sub_matrix, (m, l, -1), K1))
                remain_dic.update(top_k_pos(sub_matrix, (m, l, -1), K1, K1+1))
        
    elif mode == "random":
        total_elements = matrix.numel()
        flat_tensor = matrix.flatten()
        random_indices = torch.randperm(total_elements)[:K]
        values = flat_tensor[random_indices]
        pos = torch.unravel_index(random_indices, matrix.shape)
        pos = list(zip(*pos))
        pos = [tuple(tensor.item() for tensor in tup) for tup in pos]
        base_dic = dict(zip(values.tolist(), pos))
    
    if mode != "random":
        r_keys = sorted(remain_dic, key=remain_dic.get, reverse=True)[:remain_K]
        base_dic.update({key: remain_dic[key] for key in r_keys})

    modal_dic = {modal: [] for modal in modals}
    for value in base_dic.values():
        modality_idx = value[0]
        modal_dic[modals[modality_idx]].append(value[1:])
    return base_dic, modal_dic

def find_topK_values(data_dict, K): # no overlap
    ret = {key: [] for key in data_dict.keys()}
    
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
    return ret

def select_modality_neurons_from_importance_scores(args):
    """
    select modality-specific neurons
    """
    if not os.path.exists(args.mask_path):
        # compute weighted-sum-score
        scores = args.score_dict
        w_scores = {}

        score_weights = args.score_weights
        score_keys = args.score_keys
        assert len(score_weights) == len(score_keys)

        for modal in args.mask_modal:
            w_scores[modal] = torch.zeros(args.layer_num, args.hidden_size).to(args.device)
            for ind, key in enumerate(score_keys):
                w_scores[modal] += score_weights[ind] * scores[f'{modal}_{key}']
        # save weighted-sum-score
        with open(args.w_scores_path, 'wb') as f:
            pickle.dump(w_scores, f)
        
        # generate mask
        K = int(args.select_ratio * args.hidden_size * args.layer_num) # 5304
        L, N = args.layer_num, args.hidden_size
        if args.modality_mode == 3: # random
            k_inds = {"random": [(random.randint(0, L - 1), random.randint(0, N - 1)) for _ in range(K)]}
        else:
            for _, value in w_scores.items():
                assert torch.sum(value) != 0
            if args.selection == "adaptive":
                k_inds = find_topK_values(w_scores, K) # {'text':(L,N), ...}
                
            if args.selection == "LA_MU": # maybe overlap
                k_inds = {}
                k_split = K // len(w_scores)
                for modal, modal_scores in w_scores.items():
                    sub_dict = {modal: modal_scores}
                    k_inds.update(find_topK_values(sub_dict, k_split)) # {'text':(L,N)}
                    
            if args.selection == "LU_MA": # maybe overlap
                k_split = K // args.layer_num
                keys = list(w_scores.keys())
                Map = {i: keys[i] for i in range(len(keys))}
                k_inds = {key: [] for key in keys}
                           
                for layer in range(args.layer_num):
                    multimodal_single_layer = []
                    for modal, modal_scores in w_scores.items():
                        multimodal_single_layer.append(modal_scores[layer])
                    sub_dict = {'-': torch.stack(multimodal_single_layer)}
                    ret = find_topK_values(sub_dict, k_split)['-']
                    
                    for tuple in ret:
                        modal_ind, neuron_ind = tuple
                        k_inds[Map[modal_ind]].append((layer, neuron_ind))
                        
            if args.selection == "uniform": # best, maybe overlap
                k_inds = {m: [] for m in w_scores.keys()}
                k_split = K // (len(w_scores) * args.layer_num)
                for layer in range(args.layer_num):
                    zero_matrix = torch.zeros(args.layer_num, args.hidden_size).to(args.device)
                    for modal, modal_scores in w_scores.items():
                        w_score_layer = modal_scores[layer]
                        zero_matrix[layer] = w_score_layer
                        top_k_values = find_topK_values({modal: zero_matrix}, k_split)[modal] # {'text':(L,N)}
                        k_inds[modal].extend(top_k_values)

        # save mask
        with open(args.mask_path, 'wb') as f:
            pickle.dump(k_inds, f)
    else:
        with open(args.mask_path, 'rb') as f:
            k_inds = pickle.load(f)
                
    return k_inds

def extract_digits(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group(0))
    return False

def create_act_hook(layer_index, args):
    """
    hook for activation of mlp neurons in llm
    """
    def activation_hook(module, input, output):
        if args.mode == 0:
            return output
        
        elif args.mode == 2:
            # return output
            args.deact_val = output.min() if args.deact_val == -1 else args.deact_val
            args.w_scores_path = args.folder_path + '/weighted_score.npy'
            args.mask_path = args.folder_path + '/mask.npy'

            if not os.path.exists(args.mask_path): # load score and get top-K 
                score_path = Path(args.folder_path).parent / 'importance_scores/'
                load_files = []
                for file in score_path.rglob('*.npy'):
                    if args.ts == -1 and not extract_digits(str(file.name)):
                        load_files.append(file)
                    elif extract_digits(str(file.name)) == args.ts:
                        load_files.append(file)

                for file in load_files:
                    with open(file, 'rb') as f:
                        args.score_dict[file.stem.replace(f'_{args.ts}', '')] = pickle.load(f)[1]

            k_inds = select_modality_neurons_from_importance_scores(args)

            if len(k_inds) == 1:
                inds = next(iter(k_inds.values()))
                inds = [t[1] for t in inds if t[0] == layer_index]
                output[..., inds] = args.deact_val
            else:
                for modal in args.select_to_mask:
                    inds = k_inds[modal]
                    inds = [t[1] for t in inds if t[0] == layer_index]
                    output[..., inds] = args.deact_val

        elif output.shape[:-1] == args.modal_mask['text'].shape:
            for modal in args.mask_modal:
                mask = args.modal_mask[modal]
                dic_score = get_neuron_importance_scores_in_layer(output, mask, args)
                for score_type in args.score_keys:
                    args.score_dict[f'{modal}_{score_type}'][layer_index] += dic_score[score_type] 
        
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

def create_attn_hook(layer_index, args):
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
    args.csv_path = args.folder_path + csv_name
    args.csv_fieldnames = ["index", "dataset name", "sub-index", "text", "img", "video", "audio", "answer", "label"]
    args.csv_file = open(args.csv_path, mode='a+', newline='', encoding='utf-8')
    def initialize_csv_writer(args, add_lst):
        args.csv_fieldnames += add_lst
        args.writer = csv.DictWriter(args.csv_file, fieldnames=args.csv_fieldnames)
        args.csv_file.seek(0, 0)
        if args.csv_file.read(1) == '':
            args.csv_file.seek(0, 0)
            args.writer.writeheader()
    if args.task in ["text_vqa", "mmlu"]:
        initialize_csv_writer(args, ["correct"])
    elif args.task == "coco_caption":
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
        if args.mmlu_qa_index >= args.vqa_num:
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