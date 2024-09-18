"""class definition for Qwen2-VL"""
import torch
import numpy as np
from qwen_vl_utils import process_vision_info
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils.func import create_act_hook, create_attn_hook

from .local_packages.transformers import Qwen2VLForConditionalGeneration, AutoProcessor

class Qwen2_VL:
    """
    class definition of Qwen2-VL
    """
    def __init__(self, args):
        self.args = args
        model_path = "/home/ubuntu/models/Qwen2-VL-7B-Instruct"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

        # need to be designed manually
        self.args.layer_num = int(len(self.model.model.layers)) # 28
        self.args.hidden_size = 18944

        # use dict to store mask for all possible modals
        args.mask_dict = {key: torch.zeros(self.args.layer_num, self.args.hidden_size).to(self.model.device) for key in self.args.mask_modal}

        # register hooks
        for i in range(self.args.layer_num):
            act_hook = create_act_hook(i, self.args)
            self.model.model.layers[i].mlp.act_fn.register_forward_hook(act_hook)

            attn_hook = create_attn_hook(i, self.args)
            self.model.model.layers[i].self_attn.register_forward_hook(attn_hook)

    def infer(self, data):
        """
        inference on the given information
        """
        prompt, img_path = data.get('text'), data.get('img')
        messages = [
            {
                "role": "user",
                "content": ([{"type": "image", "image": img_path}] if img_path is not None else []) + 
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
        inputs = inputs.to(next(self.model.parameters()).device)

        # 151655: <|image_pad|>
        # 151644: <|im_start|>
        # 151645: <|im_end|>
        # 151652: <|vision_start|>
        # 151653: <|vision_end|>
        special_token = torch.tensor([151644, 151645, 151652, 151653], device='cuda')
        img_mask = inputs['input_ids'] == 151655
        spe_mask = torch.isin(inputs['input_ids'], special_token)
        text_mask = ~(img_mask | spe_mask)

        # define token mask for all possible modals
        self.args.modal_mask = {
            'text': text_mask,
            'image': img_mask,
            'special': spe_mask,
            'spe_text': spe_mask | text_mask,
            'all': torch.full_like(text_mask, True)
        }
        assert not torch.any((img_mask & spe_mask))

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]