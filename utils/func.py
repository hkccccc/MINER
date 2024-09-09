"""utils functions to assist other python files"""
import random
import torch
import os
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from transformers import CLIPProcessor, CLIPModel
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
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    
    def truncate_text(self, text, max_length):
        """
        clip text to max_length
        """
        tokens = self.clip_processor.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return self.clip_processor.tokenizer.convert_tokens_to_string(tokens)

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

    def calculate_clip_score(self, input_sentence, reference_sentences):
        """
        calculate CLIP score
        """
        max_length = self.clip_processor.tokenizer.model_max_length

        truncated_input = self.truncate_text(input_sentence, max_length)
        truncated_references = [self.truncate_text(ref, max_length) for ref in reference_sentences]

        inputs = self.clip_processor(
            text=[truncated_input] + truncated_references,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        text_features = self.clip_model.get_text_features(**inputs)
        input_feature = text_features[0]
        reference_features = text_features[1:]

        cosine_similarities = torch.nn.functional.cosine_similarity(input_feature, reference_features)
        cosine_similarities = cosine_similarities.tolist()
        result = {
            'max': max(cosine_similarities),
            'min': min(cosine_similarities),
            'mean': np.mean(cosine_similarities),
            'scores': cosine_similarities
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
            'clip_score': self.calculate_clip_score(input_sentence, reference_sentences),
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

# mudule_name need to appear in choices of args.mask_modules
def create_hook(layer_index, module_name, module_info):
    """
    hook for activation of mlp neurons
    """
    info = module_info[module_name]
    assert module_name in module_info['args'].all_modules
    def activation_hook(module, input, output):
        flat = output.view(-1, output.shape[-1])
        act_counts = (flat > 0).sum(dim=0)

        info['activation_matrix'][layer_index, :] += act_counts.cpu().numpy()
        info['token_num'] += flat.shape[0]
        
        if module_info['mask_modules'] is not None and module_name in module_info['mask_modules']:
            masks = module_info['masks']
            output[..., masks[module_name][layer_index]['neuron_index']] = 0
        return output
    return activation_hook

def safe_infer(model, data):
    """
    Compress img to avoid memory problem
    """
    img_path = data.get('img')

    if img_path is None: # no image is included
        return model.infer(data)
    else:
        while True:
            try:
                result = model.infer(data)
                if is_cp_suffix(img_path):
                    os.remove(img_path)
                return result
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    new_path = resize_image(img_path, ratio=0.75)
                    img_path = new_path
                else:
                    raise e

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

def eval_mmlu(model, subject, dev_df, test_df, qa_index, qa_num, ntrain):
    """
    answer the questions from subject one by one
    """
    print(f'next is the subject: {subject}, the qa numbers: {test_df.shape[0]}')
    cors = []
    preds = []
    k = ntrain

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        if qa_index >= qa_num:
            return np.array(cors), qa_index
        print(f'question {qa_index}:')
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        pred = safe_infer(model, {'text': prompt})

        print(f"raw pred:{pred}, pred: {pred[0]}, label:{label}")
        cor = pred[0] == label
        cors.append(cor)
        preds.append((pred, pred[0], label))
        qa_index += 1

    acc = np.mean(cors)
    cors = np.array(cors)

    print(f"Average accuracy {acc:.3f} - {subject}\n")
    print(cors)
    print(preds)
    return cors, qa_index