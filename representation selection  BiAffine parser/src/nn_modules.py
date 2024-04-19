import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
# import numpy as np
# from torch.nn import functional, init
from common import *
from flip_gradient import *
from transformers import AutoTokenizer, AutoModelForMaskedLM
from vocab import *
from bertembed import *
from bertvocab import *
from alignment import *
from alignment import Alignment, load_embeddings


class CharLSTM(torch.nn.Module):
    """
    用于处理字符级别的特征。主要用于将字符序列转换为字符级别的嵌入表示，通常在自然语言处理任务中用于构建词或词语的字符级别表示。
    """

    def __init__(self, n_char, char_dim, char_hidden, bidirectional=True):
        super(CharLSTM, self).__init__()
        self.char_embedding = torch.nn.Embedding(n_char, char_dim, padding_idx=0)
        self.bidirectional = bidirectional  # BiLSTM
        self.char_lstm = torch.nn.LSTM(input_size=char_dim, hidden_size=char_hidden, num_layers=1, \
                                       bidirectional=bidirectional, bias=True, batch_first=True)

    def forward(self, chars, chars_lengths):
        # 处理输入字符序列和生成字符级别的表示
        # chars_lengths= torch.from_numpy(chars_lengths)
        sorted_lengths, sorted_index = torch.sort(chars_lengths, dim=0, descending=True)  # 排序
        maxlen = sorted_lengths[0]
        sorted_chars = chars[sorted_index, :maxlen]  # 将字符序列按最长序列的长度进行截断，以确保输入具有相同的长度。
        # sorted_chars = Variable(torch.from_numpy(sorted_chars),requires_grad=False)
        sorted_chars = Variable(sorted_chars, requires_grad=False)
        # emb = self.char_embedding(sorted_chars.cuda())
        emb = self.char_embedding(sorted_chars)  # 使用字符嵌入层将排序后的字符序列转换为字符嵌入向量
        input = nn.utils.rnn.pack_padded_sequence(emb, sorted_lengths.cpu().numpy(), batch_first=True)  # 将向量打包
        raw_index = torch.sort(sorted_index, dim=0)[1]
        raw_index = Variable(raw_index, requires_grad=False)
        out, h = self.char_lstm(input, None)
        if not self.bidirectional:
            # 在经过LSTM层（char_lstm）后，提取隐藏状态（hidden_state）。
            # 如果bidirectional为True，则会将前向和后向LSTM的隐藏状态进行拼接，
            # 否则仅使用前向LSTM的隐藏状态。
            hidden_state = h[0]
        else:
            hidden_state = torch.unsqueeze(torch.cat((h[0][0], h[0][1]), 1), 0)
        # return torch.index_select(hidden_state.cuda(), 1, raw_index.cuda())
        # 根据排序前的原始索引，对隐藏状态进行重新排序，以将其与原始输入序列对应。
        return torch.index_select(hidden_state, 1, raw_index)


class InputLayer(nn.Module):  # 这边就是使用的预训练模型XLM-RoBERTa
    def __init__(self, name, conf, word_dict_size, ext_word_dict_size, char_dict_size, tag_dict_size,
                 ext_word_embeddings_np, bert_path, bert_dim, bert_layer, is_fine_tune=True):
        super(InputLayer, self).__init__()
        self._name = name  # 表示这是input层
        self._conf = conf
        self._word_embed = nn.Embedding(word_dict_size, conf.word_emb_dim, padding_idx=padding_id)  # 初始化形状
        """追踪"""
        self.domain_id = 0  # 添加一个成员变量以跟踪domain_id状态
        # 添加属性
        self.source_embeddings = torch.cuda.FloatTensor()
 
        self.source_subwords = []  
        self.target_subwords = []       
        self.source_counter = 0
        self.target_counter = 0
        #self.source_subword_idxs = torch.tensor([], device=self._conf.device)
        self.source_subword_idxs = torch.Tensor().cuda(self._conf.device)
        self.target_subword_idxs = torch.Tensor().cuda(self._conf.device)
        
        # 存储所有的训练过程的数据，以供在dev和test中使用
        self.all_source_embeddings =torch.Tensor().cuda(self._conf.device)
        self.all_source_subwords = []
        self.all_source_counter =  0
        # self.all_source_subword_idxs = torch.Tensor(1, 1)
        self.all_source_subword_idxs = torch.Tensor(1, 1).cuda(self._conf.device)

        #

        # self._ext_word_embed = nn.Embedding(ext_word_dict_size, conf.word_emb_dim, padding_idx=padding_id)
        # self._tag_embed = nn.Embedding(tag_dict_size, conf.tag_emb_dim, padding_idx=padding_id)

        """
        用于将词汇表中的词映射为词嵌入向量。
        在初始化时，使用了词嵌入的初始化权重，可以选择是否对这些权重进行微调
        """
        word_emb_init = np.zeros((word_dict_size, conf.word_emb_dim), dtype=data_type)  # 词嵌入向量初始化为0
        self._word_embed.weight.data = torch.from_numpy(word_emb_init)
        self._word_embed.weight.requires_grad = is_fine_tune

        hidden_size, input_size = 100, 768
        """
        用于将BERT的输出维度降维到指定的隐藏维度（hidden_size）。这是一个全连接层，用于降低维度并生成隐藏层表示
        """
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)  # 768 * 100
        weights = orthonormal_initializer(hidden_size, input_size)  # 权重初始化
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True
        b = np.zeros(hidden_size, dtype=data_type)  # 偏置初始化
        self.linear.bias.data = torch.from_numpy(b)
        self.linear.bias.requires_grad = True

        """见bertembed.py"""
        self.bert_embedding = Bert_Embedding(bert_path, bert_layer, bert_dim)

        # tag_emb_init = np.random.randn(tag_dict_size, conf.tag_emb_dim).astype(data_type) # normal distribution
        # self._tag_embed.weight.data = torch.from_numpy(tag_emb_init)
        # self._tag_embed.weight.requires_grad = is_fine_tune
        print(char_dict_size)  # 2820

        """将输入的字符序列转换为字符级别的表示。"""
        self.char_emb = CharLSTM(int(char_dict_size), 200, int(conf.tag_emb_dim / 2),
                                 True)  # char_dim=200, hidden_size=50         # 每个词对应的字符表示是先经过一层Bilstm得到的

        # self._ext_word_embed.weight.data = torch.from_numpy(ext_word_embeddings_np)
        # self._ext_word_embed.weight.requires_grad = False

        # 用于将域信息映射为域嵌入向量。这通常用于处理不同域上的文本数据，以帮助模型更好地适应不同领域的文本
        self._domain_embed = nn.Embedding(conf.domain_size + 1, conf.domain_emb_dim,
                                          padding_idx=padding_id)  # yli:19-3-26
        domain_emb_init = np.random.randn(conf.domain_size + 1, conf.domain_emb_dim).astype(data_type)  # yli:19-3-26
        self._domain_embed.weight.data = torch.from_numpy(domain_emb_init)  # yli:19-3-26
        self._domain_embed.weight.requires_grad = is_fine_tune  # yli:19-3-26

    @property
    def name(self):
        return self._name

    def _clear_source(self):
        self.source_embeddings = torch.Tensor().cuda(self._conf.device) # 每次domain id=2之后会清空source的数据
        self.source_subwords = []
        self.source_counter = 0
        self.source_subword_idxs = torch.Tensor().cuda(self._conf.device)

    def _store_source(self, source_embeddings, subwords, counter, subword_idxs):
        self.source_embeddings = source_embeddings  # 这个是降维后的bert out
        self.source_subwords = subwords
        self.source_counter = counter  
        self.source_subword_idxs = subword_idxs
    
    def _store_training_sourceall(self, source_embeddings, subwords, counter, subword_idxs, cuda_device):
        if self.all_source_counter <= 40000: #40826
            # self.all_source_embeddings = torch.cat((self.all_source_embeddings, source_embeddings), dim = 0)  # tensor(...,10,768)在第一维上拼接
            self.all_source_subwords += subwords             # list
            self.all_source_counter += counter               # int
            
            # 如果第二维度相同，直接拼接按dim=0
            if self.all_source_subword_idxs.size(1) == subword_idxs.size(1):
                self.all_source_subword_idxs = torch.cat((self.all_source_subword_idxs, subword_idxs), dim=0)
            else:
                # 找到较大的第二维度
                max_dim = max(self.all_source_subword_idxs.size(1), subword_idxs.size(1))
                min_dim = min(self.all_source_subword_idxs.size(1), subword_idxs.size(1))
                if max_dim == self.all_source_subword_idxs.size(1):
                    max_tensor = self.all_source_subword_idxs
                    min_tensor = subword_idxs
                else:
                    max_tensor = subword_idxs
                    min_tensor = self.all_source_subword_idxs

                padded_min_tensor = torch.zeros(min_tensor.size(0), max_dim, dtype=self.all_source_subword_idxs.dtype)

                padded_min_tensor[:, :min_tensor.size(1)] = min_tensor
                padded_min_tensor = padded_min_tensor.cuda(cuda_device)

                self.all_source_subword_idxs = torch.cat((max_tensor.to(torch.int64), padded_min_tensor.to(torch.int64)), dim=0) #[1030,48] 有些是填充为0
                # self.all_source_subword_idxs = torch.tensor(self.all_source_subword_idxs, dtype=torch.long)
            # self.all_source_subword_idxs = torch.cat((self.all_source_subword_idxs, subword_idxs), dim = 0)     # tensor


    def forward(self, get_vocab,mymodel, counter, myalignment, mytokenizer, subwords, words, ext_words, tags, domains, domain_id, word_lens_encoder, char_idxs_encoder, subword_idxs,
                subword_masks, token_starts_masks, sens, cuda_device):
        """
        这段代码的目的是将不同类型的嵌入（词嵌入、BERT 嵌入、字符嵌入）融合在一起

        words：词汇的输入数据，通常是整数标识的词汇索引。
        ext_words：扩展词汇的输入数据。
        tags：标签的输入数据。 
        domains：域信息的输入数据。
        domain_id:  1 ch   或   2 vi
        word_lens_encoder：编码器的输入词汇长度。
        char_idxs_encoder：编码器的字符索引。
        subword_idxs：子词索引。
        subword_masks：子词掩码。
        token_starts_masks：标记开始的掩码。
        sens：输入的句子。
        cuda_device：GPU 设备编号。
        """
        if self.training:
            if domain_id == 1:
                # if self.training == True:
                    # x_domain_embed = self._domain_embed(torch.tensor([domain_id], dtype=torch.long, device=cuda_device))
                    x_word_embed = self._word_embed(words)  # 随机初始化的词嵌入表示
                    bert_outs = self.bert_embedding(subword_idxs, subword_masks, token_starts_masks,
                                                    sens)  # 输入句子经过微调后XLM-RoBERTa得到的词嵌入表示
                        
                    """测试【可修改】把此处的向量存起来----"""
                    self._store_source(bert_outs, subwords, counter, subword_idxs)
                    # print(type(self.source_subword_idxs))
                    # print(type(self.all_source_subword_idxs))
                    self._store_training_sourceall(bert_outs, subwords, counter, subword_idxs, cuda_device)
                    print(type(self.all_source_subword_idxs),self.all_source_subword_idxs.size())  # [1030,48]
                    """-------"""
                    # print("bert_outs", bert_outs.size())
                    x_ext_word_embed = self.linear(bert_outs)  # 线性变换将 BERT 输出的维度降低到 hidden_size = 100      
                    # print("x_word_embed",x_word_embed.size())
                    if (x_ext_word_embed.size()[1] < x_word_embed.size()[1]):
                        pad = torch.zeros(x_word_embed.size()[0], x_word_embed.size()[1] - bert_outs.size()[1],
                                        x_word_embed.size()[2])  # .cuda(cuda_device)
                        pad = pad.cuda(cuda_device)
                        # print("pad",pad.size())
                        x_ext_word_embed = torch.cat((x_ext_word_embed, pad), dim=1)  # 填充

                    x_embed = x_word_embed + x_ext_word_embed  # 二者相加 = 综合的词嵌入表示
                    x_char_input = self.char_emb(char_idxs_encoder, word_lens_encoder)  # 每个词对应的字符表示是先经过一层lstm得到的
                    x_char_embed = x_char_input.view(x_embed.size()[0], x_embed.size()[1],
                                                    -1)  # x_char_embed 是通过调整 x_char_input 的形状以适应 x_embed 的维度而得到的字符嵌入。
                    # print(" x_embed size:", x_embed.size())
                    # print(" x_char_input size:", x_char_input.size())
                    x_embed = x_embed.cuda(cuda_device)
                    x_char_embed = x_char_embed.cuda(cuda_device)
                    if self.training:  # 训练时加入dropout 防止过拟合
                        x_embed, x_char_embed = drop_input_word_tag_emb_independent(x_embed, x_char_embed,
                                                                                        self._conf.emb_dropout_ratio)
                    # 二者相加再和词对应的字符表示拼接得到最终的输入
                    # print(x_embed.shape)    # [55, 12, 100]
                    # print(x_char_embed.shape)  # [55, 12, 100]
                    # print(x_domain_embed.shape)  # [1, 8]
                    #                 
                    x_final = torch.cat((x_embed, x_char_embed), dim=2) # [55, 12, 200]
                    """存储 domain_id=1 的 source embeddings"""         
                    # print("input layer最终输入到BiLSTM的X的形状：", x_final.shape)
                
            elif domain_id == 2:
                if self.source_subword_idxs is None:
                    print("source_subword_idxs is none, we will need to initialize it")
                else:
                    # x_domain_embed = self._domain_embed(torch.tensor([domain_id], dtype=torch.long, device=cuda_device))
                    x_word_embed = self._word_embed(words)  # 随机初始化的词嵌入表示 (57,10,768)
                    bert_outs = self.bert_embedding(subword_idxs, subword_masks, token_starts_masks,
                                                    sens)  # 输入句子经过微调后XLM-RoBERTa得到的词嵌入表示  (57,10,768)
                    x_ext_word_embed = self.linear(bert_outs)  # 线性变换将 BERT 输出的维度降低到 hidden_size = 100 (57,10,100)
                    
                    """对齐"""
                    # source_embeddings = self.source_embeddings                    
                    self.target_subwords = subwords             # 目标语言的子词
                    self.target_counter = counter 
                    self.target_subword_idxs = subword_idxs

                    # 对齐嵌入：使用Alignment对象的apply()方法对齐嵌入。这将生成目标语言的嵌入矩阵和相关信息
                    # bert_vocab_path = "/home/ljj/xlm-roberta-base"
                    # 可替换成对应语言的分词器
                    source_tokenizer = mytokenizer
                    target_tokenizer = mytokenizer
                    source_model = mymodel
                        
                    source_matrix = source_model.get_input_embeddings().weight[self.source_subword_idxs].detach().cpu().numpy() # (55,36,768)
                    # print("source_matrix shape:", source_matrix.shape) # (55,36,768)
                    # source_matrix = source_model.get_input_embeddings().weight.detach().numpy()  # (250002,768)
                    # 将前两个维度合并 -> [1980, 768]
                    # 将前两个维度合并成一个维度
                    source_matrix = source_matrix.reshape(-1, source_matrix.shape[-1])
                    # 打印新形状 [1980, 768]
                    # print("Reshaped source_matrix shape:", source_matrix.shape)  
                    use_subword_info = True  # 是否使用子词信息
                    max_n_word_vectors = None  # 可选参数，最大单词数量
                    neighbors = 10  # 邻居数量
                    temperature = 0.1  # 温度参数
                    target_matrix, alignment_info = myalignment.apply(
                        get_vocab,
                        self.source_subwords,self.target_subwords,self.source_counter,self.target_counter,
                        self.source_subword_idxs,self.target_subword_idxs,
                        source_tokenizer, target_tokenizer, source_matrix,
                        use_subword_info, max_n_word_vectors, neighbors, temperature
                    ) 
                    x_target_embed = torch.from_numpy(target_matrix).cuda(cuda_device) # [1254,768]
                    # print(f"target_matrix shape:{x_target_embed.shape},type:{type(x_target_embed)}") # 
                    # print(f"bert_outs shape: {bert_outs.shape},type:{type(bert_outs)}")  # [57, 10, 768]

                    # 将维度调整成和bert_outs相同
                    x_target_embed = x_target_embed.reshape(subword_idxs.shape[0], subword_idxs.shape[1], -1).cuda(cuda_device) #[57,22,768]
                        
                        
                    """---将第二维度的22子词合并成10个词的嵌入表示：1 用词的第一个子词，2 用最后一个子词， 3 用所有子词的均值---"""
                    # 先根据id判断是否是同一个词的子词，找出对应的词嵌入表示,每一个组合存在一个list中
                    sen_lens = token_starts_masks.sum(dim=1).cuda(cuda_device) # [57]标记每个句子的词个数
                    """1 用词的第一个子词"""  
                    # for e in x_target_embed[0]:  
                    #     print(e)     
                    x_target_embed = torch.split(x_target_embed[token_starts_masks], sen_lens.tolist())  # 输出是一个tuple,每个是一个句子的tensor[词数，768]
                    # for e in x_target_embed[0]:  
                    #     print(e) 
                    for i in range(len(x_target_embed)):
                        if (x_target_embed[i].size()[0] != sum(token_starts_masks[i]) and x_target_embed[i].size()[0] != len(sens[i])):
                            print("miss match bert with token strart")
                            print("x_target_embed[i]", x_target_embed[i])
                            print("sen[i]", sens[i])
                        
                    x_target_embed = pad_sequence(x_target_embed, batch_first=True).cuda(cuda_device) # vi输出是一个tensor[57,10,768]
                    # """2 用最后一个子词"""
                    # x_target_embed_method2 = x_target_embed[:, -1, :].unsqueeze(1).repeat(1, x_target_embed.size(1), 1)
                    # for e in x_target_embed_method2[0]:  
                    #     print(e) 
                    # """3 用所有子词的均值"""
                    # x_target_embed_method3 = x_target_embed.mean(dim=1, keepdim=True).repeat(1, x_target_embed.size(1), 1)
                    # for e in x_target_embed_method3[0]:  
                    #     print(e) 
                    """------"""           
                    x_target_embed = self.linear(x_target_embed).cuda(cuda_device)   # [57, 10, 768] 
                    

                    if (x_ext_word_embed.size()[1] < x_word_embed.size()[1]):
                        pad = torch.zeros(x_word_embed.size()[0], x_word_embed.size()[1] - bert_outs.size()[1],
                                        x_word_embed.size()[2])  # .cuda(cuda_device)
                        pad = pad.cuda(cuda_device)
                        # print("pad",pad.size())
                        x_ext_word_embed = torch.cat((x_ext_word_embed, pad), dim=1)  # 填充

                    if (x_target_embed.size()[1] < x_word_embed.size()[1]):
                        pad = torch.zeros(x_word_embed.size()[0], x_word_embed.size()[1] - x_target_embed.size()[1],
                                        x_word_embed.size()[2])  # .cuda(cuda_device)
                        pad = pad.cuda(cuda_device)
                        # print("pad",pad.size())
                        x_target_embed = torch.cat((x_target_embed, pad), dim=1)  # 填充

                    
                    x_embed_init = x_word_embed + x_ext_word_embed  # [57, 10, 100]
        
                    """【设置权重占比】相加"""
                    # weight_factor = 0.7
                    # x_embed = x_embed_init * weight_factor + x_target_embed * (1 - weight_factor)      # [57, 10, 100]             
                    x_embed = x_embed_init + x_target_embed# [57, 10, 100]

                    x_char_input = self.char_emb(char_idxs_encoder, word_lens_encoder)  
                    x_char_embed = x_char_input.view(bert_outs.size()[0], bert_outs.size()[1], -1)  # [57, 10, 100]

                    x_embed = x_embed.cuda(cuda_device)   # [57, 10, 100]
                    x_char_embed = x_char_embed.cuda(cuda_device) # [57, 10, 100]
                    x_target_embed = x_target_embed.cuda(cuda_device)
                    if self.training:  
                        x_embed, x_char_embed = drop_input_word_tag_emb_independent(x_embed, x_char_embed,
                                                                                        self._conf.emb_dropout_ratio)
                    x_final = torch.cat((x_embed, x_char_embed), dim=2)  # 拼接[57,10,300]
      
                    self._clear_source() 

        else:  
            # 评估阶段
            if domain_id == 1:
                # if self.training == True:
                    # x_domain_embed = self._domain_embed(torch.tensor([domain_id], dtype=torch.long, device=cuda_device))
                    x_word_embed = self._word_embed(words)  # 随机初始化的词嵌入表示
                    bert_outs = self.bert_embedding(subword_idxs, subword_masks, token_starts_masks,
                                                    sens)  # 输入句子经过微调后XLM-RoBERTa得到的词嵌入表示
                        
                    """测试【可修改】把此处的向量存起来----"""
                    self._store_source(bert_outs, subwords, counter, subword_idxs)
                    # print(type(self.source_subword_idxs))
                    # print(type(self.all_source_subword_idxs))
                    self._store_training_sourceall(bert_outs, subwords, counter, subword_idxs, cuda_device)
                    print(type(self.all_source_subword_idxs),self.all_source_subword_idxs.size())  # [1030,48]
                    """-------"""
                    # print("bert_outs", bert_outs.size())
                    x_ext_word_embed = self.linear(bert_outs)  # 线性变换将 BERT 输出的维度降低到 hidden_size = 100      
                    # print("x_word_embed",x_word_embed.size())
                    if (x_ext_word_embed.size()[1] < x_word_embed.size()[1]):
                        pad = torch.zeros(x_word_embed.size()[0], x_word_embed.size()[1] - bert_outs.size()[1],
                                        x_word_embed.size()[2])  # .cuda(cuda_device)
                        pad = pad.cuda(cuda_device)
                        # print("pad",pad.size())
                        x_ext_word_embed = torch.cat((x_ext_word_embed, pad), dim=1)  # 填充

                    x_embed = x_word_embed + x_ext_word_embed  # 二者相加 = 综合的词嵌入表示
                    x_char_input = self.char_emb(char_idxs_encoder, word_lens_encoder)  # 每个词对应的字符表示是先经过一层lstm得到的
                    x_char_embed = x_char_input.view(x_embed.size()[0], x_embed.size()[1],
                                                    -1)  # x_char_embed 是通过调整 x_char_input 的形状以适应 x_embed 的维度而得到的字符嵌入。
                    # print(" x_embed size:", x_embed.size())
                    # print(" x_char_input size:", x_char_input.size())
                    x_embed = x_embed.cuda(cuda_device)
                    x_char_embed = x_char_embed.cuda(cuda_device)
                    x_embed, x_char_embed = drop_input_word_tag_emb_independent(x_embed, x_char_embed,
                                                                                        self._conf.emb_dropout_ratio)
                    # 二者相加再和词对应的字符表示拼接得到最终的输入
                    # print(x_embed.shape)    # [55, 12, 100]
                    # print(x_char_embed.shape)  # [55, 12, 100]
                    # print(x_domain_embed.shape)  # [1, 8]
                    #                 
                    x_final = torch.cat((x_embed, x_char_embed), dim=2) # [55, 12, 200]
                    """存储 domain_id=1 的 source embeddings"""         
                    # print("input layer最终输入到BiLSTM的X的形状：", x_final.shape)
            elif domain_id == 2:    
                if self.all_source_subword_idxs is None:
                    print("all_source_subword_idxs is none, we will need to initialize it")
                else:
                    # x_domain_embed = self._domain_embed(torch.tensor([domain_id], dtype=torch.long, device=cuda_device))
                    x_word_embed = self._word_embed(words)  # 随机初始化的词嵌入表示 (57,10,768)
                    bert_outs = self.bert_embedding(subword_idxs, subword_masks, token_starts_masks,
                                                    sens)  # 输入句子经过微调后XLM-RoBERTa得到的词嵌入表示  (57,10,768)
                    x_ext_word_embed = self.linear(bert_outs)  # 线性变换将 BERT 输出的维度降低到 hidden_size = 100 (57,10,100)
                    
                    """对齐"""
                    # source_embeddings = self.source_embeddings                    
                    self.target_subwords = subwords             # 目标语言的子词
                    self.target_counter = counter 
                    self.target_subword_idxs = subword_idxs

                    # 对齐嵌入：使用Alignment对象的apply()方法对齐嵌入。这将生成目标语言的嵌入矩阵和相关信息
                    # bert_vocab_path = "/home/ljj/xlm-roberta-base"
                    # 可替换成对应语言的分词器
                    source_tokenizer = mytokenizer
                    target_tokenizer = mytokenizer
                    source_model = mymodel
                    # print(self.all_source_subword_idxs)
                    print(type(self.all_source_subword_idxs),self.all_source_subword_idxs.size())
                    source_matrix = source_model.get_input_embeddings().weight[self.all_source_subword_idxs.long()].detach().cpu().numpy() # (55,36,768)
                    # print("source_matrix shape:", source_matrix.shape) # (55,36,768)
                    # source_matrix = source_model.get_input_embeddings().weight.detach().numpy()  # (250002,768)
                    # 将前两个维度合并 -> [1980, 768]
                    # 将前两个维度合并成一个维度
                    source_matrix = source_matrix.reshape(-1, source_matrix.shape[-1])
                    # 打印新形状 [1980, 768]
                    # print("Reshaped source_matrix shape:", source_matrix.shape)  
                    use_subword_info = True  # 是否使用子词信息
                    max_n_word_vectors = None  # 可选参数，最大单词数量
                    neighbors = 10  # 邻居数量
                    temperature = 0.1  # 温度参数
                    target_matrix, alignment_info = myalignment.apply(
                        get_vocab,
                        self.all_source_subwords,self.target_subwords,self.all_source_counter,self.target_counter,
                        self.all_source_subword_idxs,self.target_subword_idxs,
                        source_tokenizer, target_tokenizer, source_matrix,
                        use_subword_info, max_n_word_vectors, neighbors, temperature
                    ) 
                    x_target_embed = torch.from_numpy(target_matrix).cuda(cuda_device) # [1254,768]
                    # print(f"target_matrix shape:{x_target_embed.shape},type:{type(x_target_embed)}") # 
                    # print(f"bert_outs shape: {bert_outs.shape},type:{type(bert_outs)}")  # [57, 10, 768]

                    # 将维度调整成和bert_outs相同
                    x_target_embed = x_target_embed.reshape(subword_idxs.shape[0], subword_idxs.shape[1], -1).cuda(cuda_device) #[57,22,768]
                                                
                    """---将第二维度的22子词合并成10个词的嵌入表示：1 用词的第一个子词，2 用最后一个子词， 3 用所有子词的均值---"""
                    # 先根据id判断是否是同一个词的子词，找出对应的词嵌入表示,每一个组合存在一个list中
                    sen_lens = token_starts_masks.sum(dim=1).cuda(cuda_device) # [57]标记每个句子的词个数
                    """1 用词的第一个子词"""  
                    # for e in x_target_embed[0]:  
                    #     print(e)     
                    x_target_embed = torch.split(x_target_embed[token_starts_masks], sen_lens.tolist())  # 输出是一个tuple,每个是一个句子的tensor[词数，768]
                    # for e in x_target_embed[0]:  
                    #     print(e) 
                    for i in range(len(x_target_embed)):
                        if (x_target_embed[i].size()[0] != sum(token_starts_masks[i]) and x_target_embed[i].size()[0] != len(sens[i])):
                            print("miss match bert with token strart")
                            print("x_target_embed[i]", x_target_embed[i])
                            print("sen[i]", sens[i])
                        
                    x_target_embed = pad_sequence(x_target_embed, batch_first=True).cuda(cuda_device) # vi输出是一个tensor[57,10,768]
                    # """2 用最后一个子词"""
                    # x_target_embed_method2 = x_target_embed[:, -1, :].unsqueeze(1).repeat(1, x_target_embed.size(1), 1)
                    # for e in x_target_embed_method2[0]:  
                    #     print(e) 
                    # """3 用所有子词的均值"""
                    # x_target_embed_method3 = x_target_embed.mean(dim=1, keepdim=True).repeat(1, x_target_embed.size(1), 1)
                    # for e in x_target_embed_method3[0]:  
                    #     print(e) 
                    """------"""           
                    x_target_embed = self.linear(x_target_embed).cuda(cuda_device)   # [57, 10, 768] 
                    
                    if (x_ext_word_embed.size()[1] < x_word_embed.size()[1]):
                        pad = torch.zeros(x_word_embed.size()[0], x_word_embed.size()[1] - bert_outs.size()[1],
                                        x_word_embed.size()[2])  # .cuda(cuda_device)
                        pad = pad.cuda(cuda_device)
                        # print("pad",pad.size())
                        x_ext_word_embed = torch.cat((x_ext_word_embed, pad), dim=1)  # 填充

                    if (x_target_embed.size()[1] < x_word_embed.size()[1]):
                        pad = torch.zeros(x_word_embed.size()[0], x_word_embed.size()[1] - x_target_embed.size()[1],
                                        x_word_embed.size()[2])  # .cuda(cuda_device)
                        pad = pad.cuda(cuda_device)
                        # print("pad",pad.size())
                        x_target_embed = torch.cat((x_target_embed, pad), dim=1)  # 填充

                    # 相加  三个
                    x_embed_init = x_word_embed + x_ext_word_embed  # [57, 10, 100]
                    """【设置权重占比】相加"""
                    #weight_factor = 0.7
                    #x_embed = x_embed_init * weight_factor + x_target_embed * (1 - weight_factor)      # [57, 10, 100]             
                    x_embed = x_embed_init + x_target_embed# [57, 10, 100]

                    x_char_input = self.char_emb(char_idxs_encoder, word_lens_encoder)  
                    x_char_embed = x_char_input.view(bert_outs.size()[0], bert_outs.size()[1], -1)  # [57, 10, 100]

                    x_embed = x_embed.cuda(cuda_device)   # [57, 10, 100]
                    x_char_embed = x_char_embed.cuda(cuda_device) # [57, 10, 100]
                    x_target_embed = x_target_embed.cuda(cuda_device)
                    if self.training:  
                        x_embed, x_char_embed = drop_input_word_tag_emb_independent(x_embed, x_char_embed,
                                                                                        self._conf.emb_dropout_ratio)
                    x_final = torch.cat((x_embed, x_char_embed), dim=2)  # 拼接[57,10,300]
                   
                    
        return x_final


class Mylinear(nn.Module):
    def __init__(self, name, input_size, hidden_size):
        super(Mylinear, self).__init__()
        self._name = name
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        weights = orthonormal_initializer(hidden_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True
        b = np.zeros(hidden_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b)
        self.linear.bias.requires_grad = True

    @property
    def name(self):
        return self._name

    def forward(self, lstm_out):
        y = self.linear(lstm_out)
        return y

        y = F.softmax(y, dim=2)
        print("softmax:", y.size())
        y = (shared_lstm_out.unsqueeze(-1) @ y.unsqueeze(-2)).sum(-1)
        print("after multiply and sum: ", y.size())
        print("shared_lstm_out", shared_lstm_out.size())
        # y = torch.mul(y, shared_lstm_out)
        return y


class EncDomain(nn.Module):
    def __init__(self, name, input_size, hidden_size, activation=None):
        super(EncDomain, self).__init__()
        self._name = name
        self.linear = nn.Linear(in_features=input_size, out_features=input_size)
        weights = orthonormal_initializer(input_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True

        self._activate = (activation or (lambda x: x))
        assert (callable(self._activate))

        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        weights1 = orthonormal_initializer(hidden_size, input_size)
        self.linear1.weight.data = torch.from_numpy(weights1)
        self.linear1.weight.requires_grad = True

    @property
    def name(self):
        return self._name

    def forward(self, shared_lstm_out):
        # y = self.linear1(self._activate(self.linear(shared_lstm_out)))
        y = self.linear1(self._activate(shared_lstm_out))
        y = F.softmax(y, dim=2)
        print("softmax:", y.size())
        y = (shared_lstm_out.unsqueeze(-1) @ y.unsqueeze(-2)).sum(-1)
        print("after multiply and sum: ", y.size())
        print("shared_lstm_out", shared_lstm_out.size())
        # y = torch.mul(y, shared_lstm_out)
        return y


class GateLSTMs(nn.Module):
    def __init__(self, name, input_size, hidden_size, activation=None):
        super(GateLSTMs, self).__init__()
        self._name = name
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        weights = orthonormal_initializer(hidden_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True
        b = np.zeros(hidden_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b)
        self.linear.bias.requires_grad = True

        self._activate = (activation or (lambda x: x))
        assert (callable(self._activate))

        self.linear1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        weights1 = orthonormal_initializer(hidden_size, input_size)
        self.linear1.weight.data = torch.from_numpy(weights1)
        self.linear1.weight.requires_grad = True

    @property
    def name(self):
        return self._name

    def forward(self, shared_lstm_out, private_lstm_out):
        # y1 = torch.cat((shared_lstm_out, private_lstm_out), dim=2)
        # g = self._activate(self.linear(y1))
        # y = torch.mul(g,shared_lstm_out) + torch.mul((1-g), private_lstm_out)
        y1 = self._activate(self.linear1(shared_lstm_out) + self.linear(private_lstm_out))
        y = torch.mul(y1, private_lstm_out)
        return y


class MLPLayer(nn.Module):
    def __init__(self, name, input_size, hidden_size, activation=None):
        super(MLPLayer, self).__init__()
        self._name = name
        self.linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        weights = orthonormal_initializer(hidden_size, input_size)
        self.linear.weight.data = torch.from_numpy(weights)
        self.linear.weight.requires_grad = True
        b = np.zeros(hidden_size, dtype=data_type)
        self.linear.bias.data = torch.from_numpy(b)
        self.linear.bias.requires_grad = True

        self._activate = (activation or (lambda x: x))
        assert (callable(self._activate))

    @property
    def name(self):
        return self._name

    def forward(self, x):
        y = self.linear(x)
        return self._activate(y)


class BiAffineLayer(nn.Module):
    def __init__(self, name, in1_dim, in2_dim, out_dim, bias_dim=(1, 1)):
        super(BiAffineLayer, self).__init__()
        self._name = name
        self._in1_dim = in1_dim
        self._in2_dim = in2_dim
        self._out_dim = out_dim
        self._bias_dim = bias_dim
        self._in1_dim_w_bias = in1_dim + bias_dim[0]
        self._in2_dim_w_bias = in2_dim + bias_dim[1]
        self._linear_out_dim_w_bias = out_dim * self._in2_dim_w_bias
        self._linear_layer = nn.Linear(in_features=self._in1_dim_w_bias,
                                       out_features=self._linear_out_dim_w_bias,
                                       bias=False)
        linear_weights = np.zeros((self._linear_out_dim_w_bias, self._in1_dim_w_bias), dtype=data_type)
        self._linear_layer.weight.data = torch.from_numpy(linear_weights)
        self._linear_layer.weight.requires_grad = True

    @property
    def name(self):
        return self._name

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size2, len2, dim2 = input2.size()
        assert (batch_size == batch_size2)
        assert (len1 == len2)
        assert (dim1 == self._in1_dim and dim2 == self._in2_dim)

        if self._bias_dim[0] > 0:
            ones = input1.new_full((batch_size, len1, self._bias_dim[0]), 1)
            input1 = torch.cat((input1, ones), dim=2)
        if self._bias_dim[1] > 0:
            ones = input2.new_full((batch_size, len2, self._bias_dim[1]), 1)
            input2 = torch.cat((input2, ones), dim=2)

        affine = self._linear_layer(input1)
        affine = affine.view(batch_size, len1 * self._out_dim, self._in2_dim_w_bias)  # batch len1*L dim2
        input2 = input2.transpose(1, 2)  # -> batch dim2 len2

        bi_affine = torch.bmm(affine, input2).transpose(1, 2)  # batch len2 len1*L; batch matrix multiplication
        return bi_affine.contiguous().view(batch_size, len2, len1, self._out_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'in1_features=' + str(self._in1_dim) \
               + ', in2_features=' + str(self._in2_dim) \
               + ', out_features=' + str(self._out_dim) + ')'


class MyLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, name, input_size, hidden_size, num_layers=1,
                 bidirectional=False, dropout_in=0, dropout_out=0, is_fine_tune=True):
        super(MyLSTM, self).__init__()
        self._name = name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        for drop in (self.dropout_in, self.dropout_out):
            assert (-1e-3 <= drop <= 1 + 1e-3)
        self.num_directions = 2 if bidirectional else 1

        self.f_cells = []
        self.b_cells = []
        for i_layer in range(self.num_layers):
            layer_input_size = (input_size if i_layer == 0 else hidden_size * self.num_directions)
            for i_dir in range(self.num_directions):
                cells = (self.f_cells if i_dir == 0 else self.b_cells)
                cells.append(nn.LSTMCell(input_size=layer_input_size, hidden_size=hidden_size))
                weights = orthonormal_initializer(4 * self.hidden_size, self.hidden_size + layer_input_size)
                weights_h, weights_x = weights[:, :self.hidden_size], weights[:, self.hidden_size:]
                cells[i_layer].weight_ih.data = torch.from_numpy(weights_x)
                cells[i_layer].weight_hh.data = torch.from_numpy(weights_h)
                nn.init.constant_(cells[i_layer].bias_ih, 0)
                nn.init.constant_(cells[i_layer].bias_hh, 0)
                for param in cells[i_layer].parameters():
                    param.requires_grad = is_fine_tune
        # properly register modules in [], in order to be visible to Module-related methods
        # You can also setattr(self, name, object) for all
        self.f_cells = torch.nn.ModuleList(self.f_cells)
        self.b_cells = torch.nn.ModuleList(self.b_cells)

    @property
    def name(self):
        return self._name

    '''
    Zhenghua: 
    in_drop_masks: drop inputs (embeddings or previous-layer LSTM hidden output (shared for one sequence) 
    shared hid_drop_masks_for_next_timestamp: drop hidden output only for the next timestamp; (shared for one sequence)
                                     DO NOT drop hidden output for the next-layer LSTM (in_drop_mask will do this)
                                      or MLP (a separate shared dropout operation)
    '''

    @staticmethod
    def _forward_rnn(cell, x, masks, initial, h_zero, in_drop_masks, hid_drop_masks_for_next_timestamp, is_backward):
        max_time = x.size(0)  # length batch dim
        output = []
        hx = (initial, h_zero)  # ??? What if I want to use an initial vector than can be tuned?
        for t in range(max_time):
            if is_backward:
                t = max_time - t - 1
            input_i = x[t]
            if in_drop_masks is not None:
                input_i = input_i * in_drop_masks
            h_next, c_next = cell(input=input_i, hx=hx)
            # padding mask
            h_next = h_next * masks[t]  # + h_zero[0]*(1-masks[t])  # element-wise multiply; broadcast
            c_next = c_next * masks[t]  # + h_zero[1]*(1-masks[t])
            output.append(h_next)  # NO drop for now
            if hid_drop_masks_for_next_timestamp is not None:
                h_next = h_next * hid_drop_masks_for_next_timestamp
            hx = (h_next, c_next)
        if is_backward:
            output.reverse()
        output = torch.stack(output, 0)
        return output  # , hx

    def forward(self, x, masks, initial=None, is_training=True):
        max_time, batch_size, input_size = x.size()
        assert (self.input_size == input_size)

        h_zero = x.new_zeros((batch_size, self.hidden_size))
        if initial is None:
            initial = h_zero

        # h_n, c_n = [], []
        for layer in range(self.num_layers):
            in_drop_mask, hid_drop_mask, hid_drop_mask_b = None, None, None
            if self.training and self.dropout_in > 1e-3:
                in_drop_mask = compose_drop_mask(x, (batch_size, x.size(2)), self.dropout_in) \
                               / (1 - self.dropout_in)

            if self.training and self.dropout_out > 1e-3:
                hid_drop_mask = compose_drop_mask(x, (batch_size, self.hidden_size), self.dropout_out) \
                                / (1 - self.dropout_out)
                if self.bidirectional:
                    hid_drop_mask_b = compose_drop_mask(x, (batch_size, self.hidden_size), self.dropout_out) \
                                      / (1 - self.dropout_out)

            # , (layer_h_n, layer_c_n) = \
            layer_output = \
                MyLSTM._forward_rnn(cell=self.f_cells[layer], x=x, masks=masks, initial=initial, h_zero=h_zero,
                                    in_drop_masks=in_drop_mask, hid_drop_masks_for_next_timestamp=hid_drop_mask,
                                    is_backward=False)

            #  only share input_dropout
            if self.bidirectional:
                b_layer_output = \
                    MyLSTM._forward_rnn(cell=self.b_cells[layer], x=x, masks=masks, initial=initial, h_zero=h_zero,
                                        in_drop_masks=in_drop_mask, hid_drop_masks_for_next_timestamp=hid_drop_mask_b,
                                        is_backward=True)
            #  , (b_layer_h_n, b_layer_c_n) = \
            # h_n.append(torch.cat([layer_h_n, b_layer_h_n], 1) if self.bidirectional else layer_h_n)
            # c_n.append(torch.cat([layer_c_n, b_layer_c_n], 1) if self.bidirectional else layer_c_n)
            x = torch.cat([layer_output, b_layer_output], 2) if self.bidirectional else layer_output

        # h_n = torch.stack(h_n, 0)
        # c_n = torch.stack(c_n, 0)

        return x  # , (h_n, c_n)
