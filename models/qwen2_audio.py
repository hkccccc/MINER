"""class definition for Qwen2-VL"""
import torch
import os
import pickle
import numpy as np
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils.func import create_activation_hook, create_attention_hook
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

class Qwen2_Audio:
    """
    class definition of Qwen2-Audio
    """
    def __init__(self, args):
        self.args = args
        model_path = "/mnt/kaichen/data/Qwen2-Audio-7B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
        self.args.device = self.model.device
        
        # need to be designed manually
        self.args.layer_num = len(self.model.language_model.model.layers) # 32
        self.args.hidden_size = self.model.config.text_config.intermediate_size # 11008

        # register hooks
        for i in range(self.args.layer_num):
            act_hook = create_activation_hook(i, self.args)
            self.model.language_model.model.layers[i].mlp.act_fn.register_forward_hook(act_hook)

            attn_hook = create_attention_hook(i, self.args)
            self.model.language_model.model.layers[i].self_attn.register_forward_hook(attn_hook)

    def infer(self, data):
        """
        inference on the given information
        """
        self.args.ISM_of_one_sample = torch.zeros(
            len(self.args.all_importance_metric_types), # T
            len(self.args.all_modalities),              # M
            self.args.layer_num,                        # L
            self.args.hidden_size,                      # N 
        ).to(self.args.device)
        
        prompt, audio_path = data.get('text'), data.get('audio')
          
        if audio_path is not None:
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ]},
            ]
        else:
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'}, 
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                ]},
            ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                # BytesIO(urlopen(ele['audio_url']).read()), 
                                ele['audio'],
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )
        
        if len(audios) == 0:
            audios = None
        
        inputs = self.processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs = inputs.to(self.args.device)

        # 151644: <|im_start|>
        # 151645: <|im_end|>
        # 151647: <|audio_bos|>
        # 151648: <|audio_eos|>
        # 151646: <|AUDIO|>
        special_token = torch.tensor([151644, 151645, 151647, 151648], device='cuda')
        audio_mask = inputs['input_ids'] == 151646
        special_mask = torch.isin(inputs['input_ids'], special_token)
        text_mask = ~(audio_mask | special_mask)
        
        # subset of ALL_MODALITIES, all possible modalities of Qwen2-VL
        self.args.prompt_mask_shape = text_mask.shape
        self.args.input_modality_masks = {
            'text': text_mask,
            'audio': audio_mask,
            'special': special_mask,
            'special_text': special_mask | text_mask,
            'prompt': torch.full_like(text_mask, True)
        }
        assert not torch.any((audio_mask & special_mask))

        # generate_ids = self.model.generate(**inputs, max_length=256)
        generate_ids = self.model.generate(**inputs, max_length=2048)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        if self.args.mode == 1:
            if not os.path.exists(self.args.mllm_dataset_ISM_path):
                os.makedirs(self.args.mllm_dataset_ISM_path)

            ISM_file_path = f"{self.args.mllm_dataset_ISM_path}/ISM.npy"
            if not os.path.exists(ISM_file_path):
                with open(ISM_file_path, "wb") as f:
                    pickle.dump((1, self.args.ISM_of_one_sample), f)
            else:
                with open(ISM_file_path, "rb") as f:
                    sample_num, current_ISM = pickle.load(f)
                sample_num += 1
                current_ISM += self.args.ISM_of_one_sample
                with open(ISM_file_path, "wb") as f:
                    pickle.dump((sample_num, current_ISM), f)
                if sample_num in self.args.all_save_sample_nums:
                    with open(f'{self.args.mllm_dataset_ISM_path}/ISM_{sample_num}.npy', "wb") as f:
                        pickle.dump((sample_num, current_ISM), f)

        return response