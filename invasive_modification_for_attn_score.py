# Since qwen2-vl/audio lacks attention matrix saving, MINER requires invasive targeted incremental code changes. The process is as follows:

# step 1: cd to the path of your transformers package
# cd /path/to/transformers/models/[qwen2_vl or qwen2]/

# step 2: Add code to save "attn_score" in the attention class
# For "qwen2_vl", modify either the "Qwen2VLFlashAttention2" or "Qwen2VLSdpaAttention" class in "qwen2_vl/modeling_qwen2_vl.py". 
# For "qwen2_audio", adjust either the "Qwen2FlashAttention2" or "Qwen2SdpaAttention" class in "qwen2/modeling_qwen2.py". 
# The specific class to modify depends on the attention mechanism you are using.


# Here’s an example with qwen2_vl; the same applies to qwen2_audio.

# Add a function to the Qwen2VLFlashAttention2 class
def get_attn_score(self, query_states, key_states):
    """
    query/key_states: torch.Size([1, 28, 1246, 128])
    return: torch.Size([1, 1246, 1246])
    self.attn_score = self.get_attn_score(query_states, key_states)
    """
    assert len(query_states.shape) == 4
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    head_dim = query_states.size(-1)
    scaling_factor = head_dim ** 0.5
    attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / scaling_factor
    attn_matrix = F.softmax(attn_scores, dim=-1)
    final_attn_matrix = attn_matrix.mean(dim=1)
    return final_attn_matrix
       
# Add code to compute the attention matrix at the end of the "forward" function
self.attn_score = self.get_attn_score(query_states, key_states)


# Add a function to the Qwen2VLSdpaAttention class
def get_attn_score(self, query_states, key_states):
    """
    query/key_states: torch.Size([1, 28, 1246, 128])
    return: torch.Size([1, 1246, 1246])
    self.attn_score = self.get_attn_score(query_states, key_states)
    """
    head_dim = query_states.size(-1)
    scaling_factor = head_dim ** 0.5
    attn_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) / scaling_factor
    attn_matrix = F.softmax(attn_scores, dim=-1)
    final_attn_matrix = attn_matrix.mean(dim=1)
    return final_attn_matrix
        
# Add code to compute the attention matrix at the end of the "forward" function    
self.attn_score = self.get_attn_score(query_states, key_states)
