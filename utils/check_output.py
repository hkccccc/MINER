import time
import os
import re

def check_stop(file_path):
    """
    if interval > 10min, we believe the py program has stopped
    """
    modification_time = os.path.getmtime(file_path)
    current_time = time.time()
    elapsed_time = current_time - modification_time
    return elapsed_time > 600

def check_single_output(txt_path):
    """
    check output txt file with txt_path
    """
    qa_num = re.match(r'(\d+)', os.path.basename(txt_path)).group(1)
    qa_num = int(qa_num)
    # count qa_num finished
    print(f'check txt: {txt_path}')
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        pattern = r'question\s\d+:\n'
        matches = re.findall(pattern, ''.join(lines))
        complete_flag = (len(matches) == qa_num)
        stop_flag = check_stop(txt_path)
            
    if complete_flag:
        print(f'the txt has finished all {qa_num} qa pairs!')
        return (1, qa_num, len(matches))
    elif not stop_flag:
        print('the python program is running!')
        return (2, qa_num, len(matches))
    else:
        print(f'the txt has finished {len(matches)} qa pairs, and {qa_num - len(matches)} pairs remain to be processed!')
        return (3, qa_num, len(matches))

def check_certain_output(mllm, task, samples, ratio, mask_modules, path='/home/ubuntu/kaichen/modality_specific/outputs'):
    """
    check output txt with certain hyperparamter settings
    """
    # find the output txt path
    root_path = f'{path}/{mllm}/{task}'
    txt_path = None
    for root, _, files in os.walk(root_path):
        for file_name in files:
            if f'mask_{"_".join(mask_modules)}_{task}' in file_name and f'{str(samples)}_{str(ratio)}' in file_name:
                txt_path = f'{root}/{file_name}'
                break
        if txt_path is not None:
            break
    check_single_output(txt_path)

def check_output():
    """
    offer a post-process function to check output files easily
    """
    root_path = input("input root output path (default is /home/ubuntu/kaichen/modality_specific/outputs): ") or '/home/ubuntu/kaichen/modality_specific/outputs'
    choice = input("wanna check all txt files(1)? or a single txt file with path(2) with hyperparameters(3)? : ")
    
    if choice == '1':
        # check all file
        complete_num, complete_lst = 0, []
        running_num, running_lst = 0, []
        stopped_num, stopped_lst = 0, []
        for root, _, files in os.walk(root_path):
            for file_name in files:
                if 'txt' in file_name:
                    txt_path = f'{root}/{file_name}'
                    status, qa_num, curr_num = check_single_output(txt_path)
                    print("")

                    if status == 1:
                        complete_num += 1
                        complete_lst.append([root, file_name, f'{curr_num}/{qa_num}'])
                    elif status == 2:
                        running_num += 1
                        running_lst.append([root, file_name, f'{curr_num}/{qa_num}'])
                    elif status == 3:
                        stopped_num += 1
                        stopped_lst.append([root, file_name, f'{curr_num}/{qa_num}'])

        total_num = complete_num + running_num + stopped_num
        print(f'total {total_num} txt files, {complete_num} complete, {running_num} running, {stopped_num} stopped')
        print('\ncomplete txt:')
        for i, value in enumerate(complete_lst):
            print(f'{i} {value[1:]}')
        print('\nrunning txt:')
        for i, value in enumerate(running_lst):
            print(f'{i} {value[1:]}')
        print('\nstopped txt:')
        for i, value in enumerate(stopped_lst):
            print(f'{i} {value[1:]}')

        l_dict = {'com': complete_lst, 'run': running_lst, 'stop': stopped_lst}
        while True:
            index = input("\ninput the txt index for detailed information(e.g. com/run/stop_1, q to quit): ")
            if index == 'q':
                break
            else:
                l_name, ind = index.split('_')
                txt_path = '/'.join(l_dict[l_name][int(ind)][:2])
                os.system(f'tail -5 {txt_path}')

    elif choice == '2':
        txt_path = input("input the absolute path of txt file: ")
        check_single_output(txt_path)

    else:
        # check single txt file
        mllm = input("input type of MLLM(default is qwen2_vl): ") or "qwen2_vl"
        task = input("input the bench task(default is text_vqa): ") or "text_vqa"
        samples = input("input num of N(default is 100): ") or '100'
        ratio = input("input num of alpha(default is 0.01): ") or '0.01'
        mask_modules = input("input modules need to be masked(default is vit_llm): ") or "vit_llm"
        
        print("")
        check_certain_output(mllm, task, int(samples), float(ratio), mask_modules.split("_"), path=root_path)

check_output()
