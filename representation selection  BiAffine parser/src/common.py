# from nn_modules import *
# from vocab import *
import torch
import numpy as np
import time

data_type = np.float32
data_type_torch = torch.float32
# data_type_int = np.int32
# data_type_int_torch = torch.int32
data_type_int = np.long
data_type_int_torch = torch.long

pseudo_word_str = '<-BOS->'
ignore_id_head_or_label = -1
ignore_domain = 0
# ignore_head_id_str = str(ignore_id_head_or_label)
padding_str = '<-PAD->'
padding_id = 0
unknown_str = '<-UNK->'
unknown_id = 1


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss, flush=True)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix', flush=True)
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(data_type))


def drop_input_word_tag_emb_independent(word_embeddings, tag_embeddings, drop_ratio):
    assert (drop_ratio >= 0.33 - 1e-5) and drop_ratio <= (0.33 + 1e-5)
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = compose_drop_mask(word_embeddings, (batch_size, seq_length), drop_ratio)
    tag_masks = compose_drop_mask(tag_embeddings, (batch_size, seq_length), drop_ratio)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale  # DO NOT understand this part.
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks
    return word_embeddings, tag_embeddings


def compose_drop_mask(x, size, drop_ratio):
    # old way (before torch-0.4)
    # in_drop_mask = x.data.new(batch_size, input_size).fill_(1 - self.dropout_in) # same device as x
    # in_drop_mask = Variable(torch.bernoulli(in_drop_mask), requires_grad=False)
    drop_mask = x.new_full(size, 1 - drop_ratio, requires_grad=False)
    return torch.bernoulli(drop_mask)
    # no need to expand in_drop_mask
    # in_drop_mask = torch.unsqueeze(in_drop_mask, dim=2).expand(-1, -1, max_time).permute(2, 0, 1)
    # x = x * in_drop_mask


def drop_sequence_shared_mask(inputs, drop_ratio):
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = compose_drop_mask(inputs, (batch_size, hidden_size), drop_ratio) / (1 - drop_ratio)
    # drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    return inputs * drop_masks  # should be broadcast-able


def compute_domain_accuray(score, true_labels):
    batch_size, len1, len2 = score.size()
    nscore = score.contiguous().view(batch_size * len1, len2)
    ntrue_labels = true_labels.view(batch_size * len1)
    total = ntrue_labels.size()[0]
    pred_labels = nscore.data.max(1)[1].cpu()
    pred_labels[ntrue_labels.eq(ignore_domain)] = ignore_domain
    correct = pred_labels.eq(ntrue_labels.cpu()).cpu().sum().item()
    ignore_word_nums = ntrue_labels.eq(ignore_domain).sum().item()
    correct = correct - ignore_word_nums
    total = total - ignore_word_nums
    accuray = correct / total
    bc_num = pred_labels.eq(1).sum().item()
    pc_num = pred_labels.eq(2).sum().item()
    pb_num = pred_labels.eq(3).sum().item()
    zx_num = pred_labels.eq(4).sum().item()
    print("predict bc, pc, pb, zx: ", bc_num, pc_num, pb_num, zx_num)
    print("domain classificay accuray, correct, total", accuray, correct, total)
    return accuray


def get_time_str():
    return time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime(time.time()))
