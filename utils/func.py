"""utils functions to assist other python files"""
import random
import os
import csv
import torch
from PIL import Image
import numpy as np
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
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2', clip_model_name='openai/clip-vit-base-patch32'):
        self.sbert_model = SentenceTransformer(model_name)

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
    coco = COCO('/home/ubuntu/datasets/coco/annotations_trainval2014/captions_val2014.json')
    root_img_path = '/home/ubuntu/datasets/coco/val2014/'
    old_imgIds = coco.getImgIds()
    with open('/home/ubuntu/datasets/coco/karparthy_split2014/coco_test.txt', 'r', encoding='utf-8') as f:
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

def resize_image(image_path, ratio=None, max_length=None):
    """
    resize img to certain size or ratio, save as xxx_cp.jpg
    """
    with Image.open(image_path) as img:
        width, height = img.size

        if width < 400 or height < 400:
            print('image too tiny! stop!')
            exit(-1)

        if max_length is not None:
            if width > height:
                new_width = max_length
                new_height = int((300 / width) * height)
            else:
                new_height = max_length
                new_width = int((300 / height) * width)
        elif ratio is not None:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
        else:
            exit(-1)
        resized_img = img.resize((new_width, new_height), Image.ANTIALIAS)

        if not is_cp_suffix(image_path):
            new_img_path = append_suffix_to_filename(image_path)
        else:
            new_img_path = image_path
        resized_img.save(new_img_path)
        return new_img_path

def get_top_acts(arr, ratio=0.01):
    """
    select the top x% neurons with highest activation frequency
    """
    num = int(arr.shape[0] * arr.shape[1] * ratio)
    flat_arr = arr.flatten()
    top_indices = np.argsort(flat_arr)[-num:]
    top_coords = np.unravel_index(top_indices, arr.shape)
    top_values = flat_arr[top_indices]
    ret = {i: {'neuron_index': [], 'activations': []} for i in range(arr.shape[0])}
    for value, (row, col) in zip(top_values, zip(*top_coords)):
        ret[row]['neuron_index'].append(col)
        ret[row]['activations'].append(value)
    return ret

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

def count_top_k_frequent_val(tensor, K):
    """
    Count the occurrence frequency of neurons corresponding to different tokens.
    """
    flat_tensor = tensor.view(-1)
    max_value = flat_tensor.max().item() + 1
    counts = torch.bincount(flat_tensor, minlength=max_value)
    sorted_indices = torch.argsort(counts, descending=True)
    top_k_indices = sorted_indices[:K]
    top_k_counts = counts[top_k_indices]
    ind, num = [], []
    for i, idx in enumerate(top_k_indices):
        ind.append(idx.item())
        num.append(top_k_counts[i].item())

    return torch.tensor(ind), torch.tensor(num)


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
    attn_val = torch.matmul(attn_score, flat_tensor) # attn1
    # attn2_val = torch.matmul(attn_score.T, flat_tensor) # attn2

    val_lst = [prob_val, mean_val, max_val, attn_val]
    assert sum(args.score_weights) == 1
    sum_val = sum(min_max_normalize(val) * weight for val, weight in zip(val_lst, args.score_weights))
    return torch.sum(sum_val, dim=0)

# def process_tensor(old_tensor, mask, args, layer_index):
#     """
#     mask top-K important neurons
#     """
#     tensor = old_tensor.clone()[mask, :]
#     ori_shape = tensor.shape
#     flat_tensor = tensor.view(-1, ori_shape[-1])

#     if args.neuron_type == 'random':
#         top_k_indices = torch.randperm(ori_shape[-1])[:args.K]
#     else:
#         # max_val
#         max_val, _ = torch.max(flat_tensor, dim=0)
#         max_val = min_max_normalize(max_val)

#         if args.neuron_type == 'sample_specific':
#             mean_val = torch.mean(flat_tensor, dim=0)
#         elif args.neuron_type in ['token_specific', 'token_group']:
#             mask = mask.squeeze(0)
#             attn_score = args.attn_score.squeeze(0)
#             attn_score = attn_score[mask, :][:, mask]

#             # attn_score = torch.softmax(attn_score, dim=1) # softmax后好像性能下降？
#             # import pdb
#             # pdb.set_trace()
#             # mean_val = torch.matmul(attn_score.T, flat_tensor) # 按照query对所有key的attn求加权和
#             mean_val = torch.matmul(attn_score, flat_tensor) # 按照所有query对某个key的attn求加权和
#         else:
#             raise ValueError("Unsupported neuron type.")
            
#         mean_val = min_max_normalize(mean_val)
#         importance = args.beta * mean_val + (1 - args.beta) * max_val
#         _, top_k_indices = torch.topk(importance, k=args.K)
#         if args.neuron_type == 'token_group':
#             top_k_indices, num = count_top_k_frequent_val(top_k_indices, args.K)
#     return top_k_indices

def create_act_hook(layer_index, args):
    """
    hook for activation of mlp neurons in llm
    """
    args.K = int(args.select_ratio * args.hidden_size)

    def activation_hook(module, input, output):
        # if args.mode == 0 or args.mode == 2:
        if args.mode == 0 or 'None' in args.mask_modal:
            return output

        args.deact_val = output.min() if args.deact_val == -1 else args.deact_val


        # generate or load importance matrix
        if output.shape[:-1] == args.modal_mask['text'].shape:
            if args.mode == 2: # save importance matrix, not apply mask
                for modal in args.mask_modal:
                    mask = args.modal_mask[modal]
                    layer_imp_val = get_neuron_importance_scores_in_layer(output, mask, args)
                    args.mask_dict[modal][layer_index] += layer_imp_val

            else: # load and apply mask
                pass
        
        # # apply mask if need?
        # for modal in args.mask_modal:
        #     output[..., module.mask_ind[modal]] = args.deact_val

        return output
    return activation_hook

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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
    # basic information
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
    if args.mode == 2:
        args.writer.writerow(csv_line)
        args.csv_file.flush()
    else:
        for key, value in csv_line.items():
            print(f'{key}: {value}')
        print("")

def softmax(x):
    """
    softmax
    """
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

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

def get_k_neurons(mask_dict, K):
    all_neurons = []
    for layer_idx, importance_tensor in mask_dict.items():
        num_neurons = importance_tensor.shape[1]
        for neuron_idx in range(num_neurons):
            importance = importance_tensor[0, neuron_idx].item()
            all_neurons.append((importance, (layer_idx, neuron_idx)))

    all_neurons_sorted = sorted(all_neurons, key=lambda x: x[0], reverse=True)
    select_neurons = []
    top_n_neurons = all_neurons_sorted[:K]
    for importance, (layer_num, neuron_idx) in top_n_neurons:
        select_neurons.append((layer_num, neuron_idx, importance))
    return select_neurons

def get_mask(N, args):
    # Step 1: 汇总所有层的神经元重要性及其全局索引
    importance_all = []
    layer_sizes = []
    start_index = 0

    for layer, importance_tensor in args.mask_dict.items():
        importance_all.append(importance_tensor.view(-1))  # 将张量展平为一维
        layer_sizes.append(importance_tensor.numel())  # 记录每一层的神经元数目
        start_index += importance_tensor.numel()

    # 拼接所有层的重要性
    importance_all = torch.cat(importance_all)

    # Step 2: 使用 torch.topk 获取全局范围内前 N 个最重要神经元及其全局索引
    topk_values, topk_global_indices = torch.topk(importance_all, N)

    # Step 3: 创建与每层相同大小的 bool 掩码，并将对应的 top-k 神经元设置为 True
    current_start = 0
    for layer, importance_tensor in args.mask_dict.items():
        # 创建与当前层形状相同的布尔掩码
        mask = torch.zeros_like(importance_tensor, dtype=torch.bool)

        # 获取当前层的索引范围
        layer_size = importance_tensor.numel()
        layer_indices = torch.arange(current_start, current_start + layer_size)

        # 找出属于当前层的神经元的全局索引
        layer_topk_indices = topk_global_indices[(topk_global_indices >= current_start) & (topk_global_indices < current_start + layer_size)] - current_start
        
        # 标记这些神经元在掩码中为 True
        if layer_topk_indices.numel() > 0:
            mask.view(-1)[layer_topk_indices] = True
        
        # 将掩码保存回字典
        args.mask_dict[layer] = mask
        
        # 更新下一层的起始索引
        current_start += layer_size
    # 检查每层的掩码
    for layer, mask in args.mask_dict.items():
        print(f"Layer {layer}: mask shape {mask.shape}, True count: {mask.sum().item()}")