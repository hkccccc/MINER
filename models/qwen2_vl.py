"""class definition for Qwen2-VL"""
import numpy as np
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import create_hook

class Qwen2_VL:
    """
    class definition of Qwen2-VL
    """
    def __init__(self, args, mask_modules=None, masks=None):
        self.masks = masks
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto",
        # )

        # default processer
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        # The default range for the number of visual tokens per image in the model is 4-16384.
        # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
        # min_pixels = 256*28*28
        # max_pixels = 1280*28*28
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

        # need to be designed manually
        self.module_info = {
            'llm': {
                'layer_num': len(self.model.model.layers), # 28
                'hidden_size': 18944,
                'activation_matrix': None,
                'token_num': 0,
            },
            'vit': {
                'layer_num': len(self.model.visual.blocks), # 32
                'hidden_size': 5120,
                'activation_matrix': None,
                'token_num': 0,
            },
            'mask_modules': mask_modules,
            'masks': masks,
            'args': args,
        }

        # initialize the activation matrix
        for module, info in self.module_info.items():
            if module in args.all_modules:
                info['activation_matrix'] = np.zeros((info['layer_num'], info['hidden_size']))

        # register hooks, need to be designed manually
        for i in range(self.module_info['vit']['layer_num']):
            hook = create_hook(i, 'vit', self.module_info)
            self.model.visual.blocks[i].mlp.act.register_forward_hook(hook)

        for i in range(self.module_info['llm']['layer_num']):
            hook = create_hook(i, 'llm', self.module_info)
            self.model.model.layers[i].mlp.act_fn.register_forward_hook(hook)

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
        inputs = inputs.to("cuda:0")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])
        return output_text[0]