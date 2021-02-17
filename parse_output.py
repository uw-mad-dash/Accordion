import os
import sys
import re
import json
import numpy as np
import math
from collections import defaultdict

def accurate_raw_numbers(file_name, out_file):
    reg = re.compile(".?Accuracy:\s(\d*)/(\d*)\s[(]\d*")
    loss_reg = re.compile(".?Loss:\s(\d*.\d*)")
    out_dict = defaultdict(list)
    with open(file_name, "r") as f:
        in_line = f.readline()
        while in_line:
            out_line = reg.findall(in_line)
            if out_line:
                out_val = out_line[0]
                out_val = [float(x) for x in out_val]
                out_dict['raw_accuracy_num'].append(out_val)
            loss_line = loss_reg.findall(in_line)
            if loss_line:
                out_val = loss_line[0]
                out_dict['Loss'].append(float(out_val))
            in_line = f.readline()
    with open(out_file, "w") as fout:
        json.dump(out_dict, fout, indent=4)

def return_accuracy(file_name):
    accuracy = list()
    with open(file_name, "r") as fout:
        accurate_pair = json.load(fout)["raw_accuracy_num"]
    for val in accurate_pair:
        correct_number = val[0]
        total_number = val[1]
        accuracy.append((correct_number*1.0)/total_number)
    return accuracy

def return_error(in_file_names):
    accu = list()
    for files in in_file_names:
        accu_out = return_accuracy(files)
        accu.append(accu_out)
    # import ipdb; ipdb.set_trace()
    accu_arr = np.array(accu)
    mean_accu = np.mean(accu_arr, axis=0)
    mean_std = np.std(accu_arr, axis=0)
    mean_st_error = 1.960 * mean_std/math.sqrt(len(in_file_names))
    return (mean_accu, mean_std, mean_st_error)

def return_comm(file_name):
    comm = list()
    with open(file_name, "r") as fin:
        comm_per_epoch = json.load(fin)
    keys = comm_per_epoch.keys()
    keys = sorted([int(x) for x in keys])
    for k in keys:
        comm.append(comm_per_epoch[str(k)])
    for i in range(1, len(comm)):
        comm[i] = comm[i] + comm[i-1]
    return comm

def return_time(file_name):
    time_taken = list()
    with open(file_name, "r") as fin:
        time_per_epoch = json.load(fin)
    keys = time_per_epoch.keys()
    keys = sorted([int(x) for x in keys])
    for k in keys:
        time_taken.append(time_per_epoch[str(k)][1]-time_per_epoch[str(k)][0])
    for i in range(1, len(time_taken)):
        time_taken[i] = time_taken[i]+time_taken[i-1]
    return (time_taken)

in_file = sys.argv[1]
parsed_file = os.path.basename(in_file).split('.')[0]+"_parsed.json"
accurate_raw_numbers(in_file, parsed_file)
timing_log_fname = os.path.basename(in_file).split('.')[0] + "_timing_log.json"
bytes_log_fname = os.path.basename(in_file).split('.')[0] + "_floats_communicated.json"
mean_err, _, _ = return_error([parsed_file])
comm_total = return_comm(bytes_log_fname)
time_total = return_time(timing_log_fname)
print  ("Final Accuracy = {}, Total Floats Comm = {}, Total Time = {}".format(
    mean_err[-1], comm_total[-1], time_total[-1]))



