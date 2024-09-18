"""utils functions to assist other python files"""
import random
import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
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
    attn_k = F.softmax(attn_score, dim=0) # dim0, softmax on column / key
    attn_val_k = torch.matmul(attn_k.T, flat_tensor)
    attn_q = F.softmax(attn_score, dim=1) # dim1, softmax on row / query
    attn_val_q = torch.matmul(attn_q, flat_tensor)

    keys = ['prob', 'mean', 'max', 'attn_k', 'attn_q']
    values = [prob_val, mean_val, max_val, torch.sum(attn_val_k, dim=0), torch.sum(attn_val_q, dim=0)]
    return {key: min_max_normalize(value) for key, value in zip(keys, values)}

def create_act_hook(layer_index, args):
    """
    hook for activation of mlp neurons in llm
    """
    args.K = int(args.select_ratio * args.hidden_size)

    def activation_hook(module, input, output):
        if args.mode == 0:
            return output
        elif args.mode == 2:
            pass # load and apply

        args.deact_val = output.min() if args.deact_val == -1 else args.deact_val
        if output.shape[:-1] == args.modal_mask['text'].shape:
            for modal in args.mask_modal:
                mask = args.modal_mask[modal]
                dic_score = get_neuron_importance_scores_in_layer(output, mask, args)
                for score_type in dic_score.keys():
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
    if args.mode == 1:
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