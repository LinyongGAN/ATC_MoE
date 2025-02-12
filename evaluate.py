import os
import numpy as np
import torch
import torch.nn.functional as F
from transformer.ATC import Model
import soundfile as sf
from train import nopeak_mask
from transformer.process import padding_zero
import math
import fairseq

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_vars(src, model):
    src_mask = None
    e_output = model.encoder(src, src_mask).to(device)
    outputs = torch.LongTensor([[7]]).to(device)
    trg_mask = nopeak_mask(1).to(device)
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(3)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    outputs = torch.zeros(3, 200, device=device).long()
    outputs[:, 0] = 7 # init_token
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(3, e_output.size(-2), e_output.size(-1), device=device)
    e_outputs[:, :] = e_output[0]
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    # 对out进行取top_k的操作，需要注意的是这里相当于对三个句子的下一个token进行top_k的取值
    probs, ix = out[:, -1].data.topk(k)
    # 对每个可能的值都进行概率对累加
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    # 根据概率累加的结果，把三句话分别对三个beam拉直后找到概率最大的那个 至于这里为什么要进行概率的累加，在扩充的部分会讲到
    k_probs, k_ix = log_probs.view(-1).topk(k)
    # 找出topk的那几个k_probs对应在ix中的位置 row中存分别在哪几行 col存在哪几列
    row = k_ix // k
    col = k_ix % k
    # 这一步是从top_k*top_k个候选中挑出概率最大的top_k个 所以这个地方实际上还是一个3*max_length的维度
    outputs[:, :i] = outputs[row, :i]
    # 将选出来的序列后面接上概率最大的索引
    outputs[:, i] = ix[row, col]
    # 将概率值升维后进行返回
    log_scores = k_probs.unsqueeze(0)
    # 将得到的分数内容进行返回
    return outputs, log_scores

def beam_search(input, model):
    outputs, e_outputs, log_scores = init_vars(input, model)
    eos_tok = 8
    # 此处稍微有些冗余，可以考虑这是第二遍生成mask矩阵（一模一样的内容），可以进行优化
    src_mask = None
    ind = None
    # 这里的第二个参数是最大输出长度
    for i in range(2, 200):
        # 此处因为已经有<sos>以及<sos>结合e_outputs得到的第二个token索引，所以编码需要从2开始
        trg_mask = nopeak_mask(i).to(device)
        # 使用encoder编码的信息e_outputs结合当前out中的内容再预测一波
        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
        # 将得到的结果进行softmax（Transformer论文原文的最后一步）
        out = F.softmax(out, dim=-1)
        # 这里面做的事情实际上就是每次对top_k的变量进行调整 outputs是一个(3,50)的结果向量，log_scores是
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, 3)
        # 这一步是为了能够找到输出矩阵中eos的位置坐标
        ones = (outputs == eos_tok).nonzero()
        # sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
        # 如果outputs中有存在预测出的eos
        for vec in ones:
            # i在这里是第一个的坐标，实际上就是滴几句话的意思
            i = vec[0]
            # 这里判断的意思是，如果当前这句话的句子长度本身就是0
            if sentence_lengths[i] == 0:
                # 将第一个结束字符的位置给到length
                sentence_lengths[i] = vec[1]  # Position of first end symbol
        # 统计一下beam=3的情况下已经结束了的句子的个数
        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        # 如果这三个句子都已经完全结束了 通过beamsize个分数的分数来找一句最可能的
        if num_finished_sentences == 3:
            alpha = 0.7
            # 将每句话叠加概率进行计算，找到最可能的那句话
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            # 找到概率最大的那个
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    # 首先将词典进行反转
    # 如果没有找到合适的那个ind
    length = 200
    print(outputs)
    if ind is None:
        # 找到结尾的那个eos_tok
        #length = (outputs[0] == eos_tok).nonzero()[0]
        #length = (outputs[0] == eos_tok).nonzero()
        # 因为第一个是开始的那个字符，所以需要跳过第一个
        return outputs[0, 1:length]
    else:
        return outputs[ind, 1:length]

def fetch_audio(vectorization_path, audio_path):
    vectorization_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([vectorization_path])
    vectorization_model = vectorization_model[0]
    vectorization_model.eval()
    X, _ = sf.read(audio_path)
    X_pad = padding_zero(X, 64600) # padding & tensorlize
    X_tensor = torch.from_numpy(X_pad).float().reshape(1, -1)
        
    input = vectorization_model.feature_extractor(X_tensor).transpose(1,2) # audio
    model = Model()
    file_path = "/mnt/workspace/ganlinyong/transformer_audio_classification/ckpt/40_ckpt_loss_7.423316593538656e-05.pth"
    model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))
    model.eval()
    with torch.no_grad():
        out = beam_search(input.to(device), model)
    
def eval(vectorization_path):
    audio_path = "/mnt/workspace/ganlinyong/test_3.flac"
    fetch_audio(vectorization_path, audio_path)
        

if __name__ == "__main__":
    eval(vectorization_path="/mnt/workspace/ganlinyong/fusion_model_test/wav2vec_small.pt")