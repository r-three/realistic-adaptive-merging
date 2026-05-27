import re


def extract_params(s):

    p_step = r'Train loader length is ([0-9.]+)'
    p_bs = r'batch_size ([0-9.]+)'
    p_effective_bs = r'effective batch size ([0-9.]+)'
    
    # Extracted loss value
    step = int(re.search(p_step, s).group(1))
    batch_size = int(re.search(p_bs, s).group(1))
    effective_batch_size = int(re.search(p_effective_bs, s).group(1))
    
    return step, batch_size, effective_batch_size


def extract_values(s):

    p_batch_num = r'batch # ([0-9.]+)'
    p_curr_loss = r'curr loss:\s([0-9.]+)'
    p_running_loss = r'running loss:\s([0-9.]+)'
    p_acc_count = r'acc count:\s([0-9.]+)'
    p_running_acc = r'running acc:\s([0-9.]+)'
    
    # Extracted loss value
    batch_num = float(re.search(p_batch_num, s).group(1))
    curr_loss = float(re.search(p_curr_loss, s).group(1))
    running_loss = float(re.search(p_running_loss, s).group(1))
    acc_count = float(re.search(p_acc_count, s).group(1))
    running_acc = float(re.search(p_running_acc, s).group(1))
    
    return batch_num, curr_loss, running_loss, acc_count, running_acc


def extract_epoch_vals(s):

    p_epoch = r'Epoch ([0-9.]+)'
    p_loss = r'Loss:\s([0-9.]+)'
    p_acc = r'Accuracy[:\s]+([0-9.]+)'

    epoch = int(re.search(p_epoch, s).group(1))
    loss = float(re.search(p_loss, s).group(1))
    acc = float(re.search(p_acc, s).group(1))
    
    return epoch, loss, acc


def extract_lr(s):
    
    p_batch_num = r'batch number ([0-9.]+)'
    
    lr = float(s.split(' ')[-1])
    batch_num = float(re.search(p_batch_num, s).group(1))
    
    return batch_num, lr