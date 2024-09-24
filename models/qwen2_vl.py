"""class definition for Qwen2-VL"""
import torch
import os
import pickle
import numpy as np
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils.func import create_activation_hook, create_attention_hook

class Qwen2_VL:
    """
    class definition of Qwen2-VL
    """
    def __init__(self, args):
        self.args = args
        model_path = "/mnt/kaichen/data/Qwen2-VL-7B-Instruct"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.args.device = self.model.device
        # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        # need to be designed manually
        self.args.layer_num = self.model.config.num_hidden_layers # 28
        self.args.hidden_size = self.model.config.intermediate_size # 18944

        # register hooks
        for i in range(self.args.layer_num):
            act_hook = create_activation_hook(i, self.args)
            self.model.model.layers[i].mlp.act_fn.register_forward_hook(act_hook)

            attn_hook = create_attention_hook(i, self.args)
            self.model.model.layers[i].self_attn.register_forward_hook(attn_hook)

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
        
        prompt, img_path, video_path = data.get('text'), data.get('img'), data.get('video')
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image", "image": img_path}] if img_path is not None else []) + 
                            ([{"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0,}] if video_path is not None else []) + 
                            [{"type": "text", "text": prompt}],
            }
        ]
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.args.device)

        # 151655: <|image_pad|>
        # 151656: <|video_pad|>
        # 151644: <|im_start|>
        # 151645: <|im_end|>
        # 151652: <|vision_start|>
        # 151653: <|vision_end|>
        special_token = torch.tensor([151644, 151645, 151652, 151653], device='cuda')
        image_mask = inputs['input_ids'] == 151655
        video_mask = inputs['input_ids'] == 151656
        special_mask = torch.isin(inputs['input_ids'], special_token)
        text_mask = ~(image_mask | special_mask | video_mask)
        
        # subset of ALL_MODALITIES, all possible modalities of Qwen2-VL
        self.args.prompt_mask_shape = text_mask.shape
        self.args.input_modality_masks = {
            'text': text_mask,
            'image': image_mask,
            'video': video_mask,
            'special': special_mask,
            'special_text': special_mask | text_mask,
            'prompt': torch.full_like(text_mask, True)
        }
        assert not torch.any((image_mask & special_mask))

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
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

        return output_text[0]