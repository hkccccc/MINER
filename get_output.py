import ast
import pickle
from pathlib import Path
import pandas as pd

def count_layer_num(input_list):
    from collections import defaultdict
    result = defaultdict(list)

    for key, value in input_list:
        result[key].append(value)

    result = dict(sorted(result.items()))
    num_result = {key: len(result[key]) for key in result.keys()}
    return result, num_result

output_path = Path('/mnt/kaichen/radar_onellm/modality_specific/outputs')
output_path = Path('/mnt/kaichen/radar_onellm/modality_specific/demo_outputs')

def get_outputs(file_str=None):
    for file in output_path.rglob('*.csv'):
        contain_flag = False
        if file_str is not None:
            for Str in file_str:
                if Str in str(file):
                    contain_flag = True
                    break
            if not contain_flag:
                continue
        
        df = pd.read_csv(file)
        print(len(df), str(file.stem).split('--'))
        ret = {"correct": [], "bleu": [], "sbert_similarity": [], "cider": [], "wrr": []}
        for index, row in df.iterrows():
            for key in ret.keys():
                if key in row:
                    ret[key].append(row[key])
        if len(ret["correct"]) != 0:
            print("correct", sum(ret["correct"]) / len(ret["correct"]))
        elif len(ret["wrr"]) != 0:
            print("wrr", sum(ret["wrr"]) / len(ret["wrr"]))
        else:
            for key in ["bleu", "sbert_similarity", "cider"]:
                dic = {'max': [], 'min': [], 'mean': []}
                for i in range(len(df)):
                    single_dic = ast.literal_eval(ret[key][i])
                    for key2 in dic.keys():
                        dic[key2].append(single_dic[key2])
                for key2 in dic.keys():
                    print(key, key2, sum(dic[key2]) / len(df))
        print("")
        
def check_masks(file_str=None):
    for file in output_path.rglob('*mask.npy'):
        contain_flag = False
        if file_str is not None:
            for Str in file_str:
                if Str in str(file):
                    contain_flag = True
                    break
            if not contain_flag:
                continue

        print(file)
        with open(file, 'rb') as f:
            mask_dic = pickle.load(f)
            for modal in mask_dic.keys():
                print(modal, len(mask_dic[modal]))
                print(count_layer_num(mask_dic[modal])[1])
        print("")

# get_outputs(["layer_uniform"])
# get_outputs(["LA_MU"])
get_outputs()