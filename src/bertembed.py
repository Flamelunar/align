import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
# from pytorch_pretrained_bert.modeling import BertModel
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scalarmix import ScalarMix


class Bert_Embedding(nn.Module):
    def __init__(self, bert_path, bert_layer, bert_dim, freeze=True):
        super(Bert_Embedding, self).__init__()
        self.bert_layer = bert_layer
        # self.bert = BertModel.from_pretrained(bert_path)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        self.scalar_mix = ScalarMix(bert_dim, bert_layer)

        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks, sens): #[57,22]
        sen_lens = token_starts_masks.sum(dim=1)  # 句子中的单词个数
        final_outs, hidden_outs = self.bert(
            subword_idxs,#[57,22]
            token_type_ids=None,
            attention_mask=subword_masks,
            output_hidden_states=True
            #            output_all_encoded_layers=True,
        )
        # print(final_outs)
        # print("final outs",final_outs.shape)#[180,33,250002]
        # bert_outs=hidden_outs[0]
        # print("bert outs",bert_outs.shape)#[180,33,768]
        # print("hidden_outs")
        # print(hidden_outs)
        # print("len(hidden_outs)")
        # print(len(hidden_outs))#13层
        # print("hidden_outs[-1]")
        # print(hidden_outs[-1])
        bert_outs = hidden_outs #13层
        # output_all_encoded_layers=True,
        bert_outs = bert_outs[len(bert_outs) - self.bert_layer: len(bert_outs)]  # 获取后四层的表示，这个是一个list，里面存放了所有层的表示
        # print("embed bert outs bert_outs[0].size()",bert_outs[0].size())
        # rint("embed bert outs",bert_outs.shape)
        bert_outs = self.scalar_mix(bert_outs)  # 输入是一个list里面有四个[3,10,768],输出是一个[3,10,768]  vi[57,22,768]
        # print("embed after scalar mix",bert_outs.size())
        bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())  # 输出是一个tuple,每个是一个句子的tensor[词数，768]
        # print(bert_outs[token_starts_masks])
        for i in range(len(bert_outs)):
            if (bert_outs[i].size()[0] != sum(token_starts_masks[i]) and bert_outs[i].size()[0] != len(sens[i])):
                print("miss match bert with token strart")
                print("bert_outs[i]", bert_outs[i])
                print("sen[i]", sens[i])
        # print("embed after split bert outs len ",bert_outs)
        # print("embed after split bert outs ",bert_outs[0].size())#[4, 768]
        # print("embed after split bert outs ",bert_outs[1].size())#[3, 768]
        # print("embed after split bert outs ",bert_outs.size())
        bert_outs = pad_sequence(bert_outs, batch_first=True) # vi输出是一个tensor[55,10,768]
        # print("embed after pad bert outs ",bert_outs)
        # print("embed after pad bert outs",bert_outs.size())  
        # [55, 12, 768]  [55, 12, 768] 表示 BERT 模型已处理了一个批次的 12 个序列，每个序列长度为 55(每个长度又是一个句子)，每个标记的输出是一个大小为 768 的向量。

        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False


class Bert_Encoder(nn.Module):
    def __init__(self, bert_path, bert_dim, freeze=False):
        super(Bert_Encoder, self).__init__()
        self.bert_dim = bert_dim
        # self.bert = BertModel.from_pretrained(bert_path)
        self.bert = AutoModelForMaskedLM.from_pretrained(bert_path)
        
        if freeze:
            self.freeze()

    def forward(self, subword_idxs, subword_masks, token_starts_masks):
        sen_lens = token_starts_masks.sum(dim=1)
        bert_outs, _ = self.bert(
            subword_idxs,
            token_type_ids=None,
            attention_mask=subword_masks,
            output_all_encoded_layers=False,
        )
        bert_outs = torch.split(bert_outs[token_starts_masks], sen_lens.tolist())
        bert_outs = pad_sequence(bert_outs, batch_first=True)
        return bert_outs

    def freeze(self):
        for para in self.bert.parameters():
            para.requires_grad = False
