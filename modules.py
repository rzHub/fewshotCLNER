import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import *
from transformers import BertConfig, BertModel, AdamW, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
# from transformers import (
#     AutoConfig,
#     AutoModelForMaskedLM,
#     get_scheduler,
#     SchedulerType,)
from our_models import BertConfig, BertForMaskedLM
import logging
from viterbi import ViterbiDecoder
logger = logging.getLogger(__name__)

# Below 2 functions are for task embedding, not used in this published paper.
def gumbel_sigmoid_oldbug(logits, tau=2 / 3, hard=True, use_gumbel=True, generator=None):
    """gumbel-sigmoid estimator"""
    # tau = 1
    # tau = 0.1
    # tau = 1/3
    # tau=1/3
    # 从指数分布中抽样 exponential_() 能被seed控制
    if use_gumbel:
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=generator).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.sigmoid()
    else:  # 不要gumbel采样了
        y_soft = (logits / tau).sigmoid()
    if hard:
        # Straight through.
        y_hard = (y_soft > 0.5).float()
        ret = y_hard - y_soft.detach() + y_soft  # 用了hard之后激活是0的logits梯度也不会得到梯度
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret  # if hard = True then binary else continuous


# bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
# gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
def gumbel_sigmoid(logits, tau=2 / 3, hard=True, use_gumbel=True, dim=-1, generator=None):
    tau = 2 / 3
    # sigmoid to softmax
    # print(logits.shape)
    # print((logits>=0.).float().sum(-1))
    # print(logits.sigmoid().mean(-1) * 768)
    # print(logits.sigmoid().mean(-1))
    # print(torch.sum(torch.relu(logits))/torch.sum(logits>=0.))
    # print(-torch.sum(torch.relu(-logits))/torch.sum(-logits>=0.))
    logits = torch.stack([logits, torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)], dim=-1)

    if use_gumbel:
        gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_(generator=generator).log()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        # gumbels = (logits/tau + gumbels)  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
    else:
        y_soft = (logits / tau).softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    ret = ret[..., 0]
    # print(ret.sum(-1))
    return ret


def transpose_for_scores(x, num_heads, head_size):
    """ split head """
    # x: [bat,len,totalhid]
    new_x_shape = x.size()[:-1] + (num_heads, head_size)  # [bat,len,num_ent,hid]
    # new_x_shape = x.size()[:-1] + (self.num_ent*2 + 1, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)  # [bat,num_ent,len,hid]


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """ mask 句子非pad部分为 1"""
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    if lengths.is_cuda:
        row_vector = row_vector.cuda()
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask


def count_params(model_or_params: Union[torch.nn.Module, torch.nn.Parameter, List[torch.nn.Parameter]],
                 return_trainable=True, verbose=True):
    """
    NOTE: `nn.Module.parameters()` return a `Generator` which can only been iterated ONCE.
    Hence, `model_or_params` passed-in must be a `List` of parameters (which can be iterated multiple times).
    """
    if isinstance(model_or_params, torch.nn.Module):
        model_or_params = list(model_or_params.parameters())
    elif isinstance(model_or_params, torch.nn.Parameter):
        model_or_params = [model_or_params]
    elif not isinstance(model_or_params, list):
        raise TypeError("`model_or_params` is neither a `torch.nn.Module` nor a list of `torch.nn.Parameter`, "
                        "`model_or_params` should NOT be a `Generator`. ")

    num_trainable = sum(p.numel() for p in model_or_params if p.requires_grad)
    num_frozen = sum(p.numel() for p in model_or_params if not p.requires_grad)

    if verbose:
        logger.info(f"The model has {num_trainable + num_frozen:,} parameters, "
                    f"in which {num_trainable:,} are trainable and {num_frozen:,} are frozen.")

    if return_trainable:
        return num_trainable
    else:
        return num_trainable + num_frozen


def check_param_groups(model: torch.nn.Module, param_groups: list, verbose=True):
    # grouped_params_set = set()
    # for d in param_groups:
    #     for p in d['params']:
    #         grouped_params_set.add(id(p))
    # # assert grouped_params_set == set([id(p) for p in model.parameters()])
    # is_equal = (grouped_params_set == set([id(p) for p in model.parameters()]))

    num_grouped_params = sum(count_params(group['params'], verbose=False) for group in param_groups)
    num_model_params = count_params(model, verbose=False)
    is_equal = (num_grouped_params == num_model_params)

    if verbose:
        if is_equal:
            logger.info(f"Grouped parameters ({num_grouped_params:,}) == Model parameters ({num_model_params:,})")
        else:
            logger.warning(f"Grouped parameters ({num_grouped_params:,}) != Model parameters ({num_model_params:,})")
    return is_equal


class NerModel(torch.nn.Module):
    def save_model(self, path, info=''):
        torch.save({
            'state_dict': self.state_dict(),
            'opt': self.opt.state_dict(),
        }, path)
        logger.info(f'[{info}] Saved Model: {path}')

    def load_model(self, path, info='', **kwargs):
        if hasattr(self.args, 'device'):
            map_location = self.args.device
        else:
            map_location = None
        dct = torch.load(path, **kwargs, map_location=map_location)
        self.load_state_dict(dct['state_dict'])
        self.opt.load_state_dict(dct['opt'])
        logger.info(f'[{info}] Loaded Model: {path}')

    def init_opt(self):
        no_decay = ['bias', 'LayerNorm.weight']
        self.grouped_params = [
            {
                "params": [p for n, p in self.bert_layer.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.bert_layer.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        if hasattr(self, 'grouped_params'):
            params = self.grouped_params
        else:
            params = self.parameters()
        self.opt = torch.optim.AdamW(params, lr=self.lr)  # default weight_decay=1e-2
        # self.opt = AdamW(params, lr=self.lr)  # Transformer impl. default weight_decay=0.
        count_params(self)

    def init_lrs(self, num_step_per_epo=None, epo=None, num_warmup_steps=None):
        if epo is None:
            epo = self.args.num_epochs
        if num_step_per_epo is None:
            # num_step_per_epo = (num_training_instancs - 1) // self.args.batch_size + 1
            num_step_per_epo = 1234
        num_training_steps = num_step_per_epo * epo
        if num_warmup_steps is None:
            ratio = 0.1
            num_warmup_steps = ratio * num_training_steps
        # print(num_training_instancs, epo, ratio, num_step_per_epo, num_training_steps)
        self.lrs = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # self.lrs = get_scheduler(
        #     name=self.args.lr_scheduler_type,
        #     optimizer=self.opt,
        #     num_warmup_steps=self.args.warmup_step,
        #     num_training_steps=num_training_steps,
        # )    # TODO,这两个self.lrs的效果是一样的，建议到时候改为第一个（spankl原始的），因为不用添加很多args.lr_scheduler_type等参数

class BaselineExtend(NerModel):
    def __init__(self, args, loader):
        super(BaselineExtend, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.grad_clip = None
        if args.corpus == 'onto' or args.corpus == 'onto_fs':
            self.grad_clip = 1.0
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        self.use_bert = args.pretrain_mode == 'fine_tuning'

        if self.use_bert:
            self.dropout_layer = nn.Dropout(p=args.enc_dropout)
            self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
            self.bert_layer = BertModel.from_pretrained(args.bert_model_dir)
            encoder_dim = self.bert_conf.hidden_size
            # self.lr = 5e-5
            # self.lr = 1e-4
            self.lr = self.args.bert_lr  # 5e-5
        else:
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )
            encoder_dim = 1024
            self.lr = self.args.lr  # 1e-3

        num_total_ents = sum(self.num_ents_per_task)

        # Extend NER
        self.ent_layer = torch.nn.Linear(encoder_dim, 1 + num_total_ents * 2)  # B I for each ent, O for each task  ExtendNER
        # print('total_params:', sum(p.numel() for p in self.parameters()))
        # print(*[n for n, p in self.named_parameters()], sep='\n')

        self.task_offset_lst = []
        offset_s = 1
        offset_e = None
        for num_ents in self.num_ents_per_task:
            if not self.task_offset_lst:
                offset_e = offset_s + num_ents * 2
                self.task_offset_lst.append([offset_s, offset_e])
            else:
                offset_s = offset_e
                offset_e = offset_s + num_ents * 2
                self.task_offset_lst.append([offset_s, offset_e])

        # self.task_offset_lst = [[1,5], [5, 9]]  # e.g. 2ent per task  ent2id: [O, B-e1, I-e1, B-e2, I-e2, B-e3, I-e3, B_e4, I-e4]
        print('task_offset_lst (task offset in ent_layer dim)', self.task_offset_lst)  # e.g. fewnerd [[0, 15], [15, 32], [32, 53], [53, 78], [78, 97], [97, 114], [114, 127], [127, 140]]
        # print('offset_split (num_dim in ent_layer per task)', self.offset_split)  # e.g. fewnerd   # [15, 32, 53, 78, 97, 114, 127]
        # print('taskid2tagid_range', self.taskid2tagid_range)  # e.g. fewnerd {0: [1, 14], 1: [15, 30], 2: [31, 50], 3: [51, 74], 4: [75, 92], 5: [93, 108], 6: [109, 120], 7: [121, 132]}

        self.ce_loss_layer = nn.CrossEntropyLoss(reduction='none')
        count_params(self)

        if self.use_bert:
            fast_lr = self.args.lr  # 1e-3
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            entlayer_p_weight = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'weight' in n]
            entlayer_p_bias = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'bias' in n]
            p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and any(nd in n for nd in no_decay)]
            p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and not any(nd in n for nd in no_decay)]
            self.grouped_params = [
                {'params': entlayer_p_weight, 'lr': fast_lr},
                {'params': entlayer_p_bias, 'weight_decay': 0.0, 'lr': fast_lr},
                {'params': p2, 'weight_decay': 0.0},
                {'params': p3},
            ]
            if p1:  # using task emb
                self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': fast_lr}] + self.grouped_params

            check_param_groups(self, self.grouped_params)

        self.init_opt()
        if self.use_schedual:
            self.init_lrs()

    def encode(self, inputs_dct):
        batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
        seq_len = inputs_dct['ori_seq_len']

        if self.use_bert:
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)

            bert_out = self.dropout_layer(bert_out)  # don't forget
            encode_output = bert_out

        else:  # bilstm
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            encode_output = rnn_output_x

        # ffn
        encode_output = self.ent_layer(encode_output)
        self.batch_tag_tensor = encode_output
        return encode_output

    def decode(self, ent_output, seq_len, task_id):
        offset_s, offset_e = self.task_offset_lst[task_id]

        mask = sequence_mask(seq_len, dtype=torch.uint8)
        # decode_ids = self.crf_layer.decode(ent_output, mask)

        ent_output_prob = ent_output[:, :, :offset_e].softmax(-1)

        id2tag = self.loader.datareader.id2tag
        curr_task_id2tag = {i: t for i, t in id2tag.items() if i < offset_e}
        curr_task_tag2id = {t: i for i, t in curr_task_id2tag.items()}
        curr_task_num_tags = len(curr_task_id2tag)
        trans_mask = get_BIO_transitions_mask(curr_task_tag2id, curr_task_id2tag)
        trans_mask = torch.tensor(1 - trans_mask).to(ent_output_prob.device)
        num_valid_for_prev_tag = trans_mask.sum(-1)
        trans_mask = trans_mask / num_valid_for_prev_tag
        start_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)
        end_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)

        decode_ids = viterbi_decode(ent_output_prob, mask, start_transitions, end_transitions, trans_mask)
        return decode_ids

    def calc_ce_loss(self, target, predict, seq_len_mask, ce_mask):
        # target [b,l]  predict [b,l,tag]
        ce_loss = self.ce_loss_layer(predict.transpose(1, 2), target)  # [b,ent,l] [b,l] -> [b,l]
        ce_loss = ce_loss * seq_len_mask
        ce_loss = ce_loss * ce_mask
        # return span_loss。mean()  # [b,l]

        num_ce = torch.sum(torch.logical_and(seq_len_mask, ce_mask))
        if num_ce == 0.:
            ce_loss = 0.
        else:
            ce_loss = ce_loss.sum() / num_ce
        # print(num_ce, ce_loss)
        return ce_loss  # [b,l]

    def calc_kl_loss(self, predict, target, seq_len_mask, kl_mask, ofs, ofe):
        # kl_loss = torch.nn.functional.kl_div(log_kl_pred.double(), target.double(), reduction='none')  # [num_spans, ent, 2]
        # target = target / 2
        # print('target', target)
        # print('predict', predict)
        # target_prob = target.softmax(dim=-1)
        # print('target_prob', target.softmax(dim=-1))
        # print('predic_prob', predict.softmax(dim=-1))
        # curr_ent_dim = ofe - ofs
        # pad_tenosr = torch.zeros_like(target_prob)[:, :, :curr_ent_dim]
        # target_prob = torch.cat([target_prob, pad_tenosr], dim=-1)

        # kl_loss = torch.nn.functional.kl_div(predict.softmax(dim=-1).log(), target_prob, reduction='none')  # [b,l,dim]
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(predict, dim=-1),
            torch.nn.functional.log_softmax(target, dim=-1),
            log_target=True,
            reduction='none')  # [b,l,dim]

        kl_loss = kl_loss.sum(-1)  # [b,l]
        kl_loss = kl_loss * seq_len_mask
        kl_loss = kl_loss * kl_mask

        num_kl = torch.sum(torch.logical_and(seq_len_mask, kl_mask))
        if num_kl == 0.:
            kl_loss = 0.
        else:
            kl_loss = kl_loss.sum() / num_kl
        # print(num_kl, kl_loss)
        # print(seq_len_mask.sum())
        return kl_loss  # [b,l]

    def calc_loss(self, ce_target, kl_target, ent_output, seq_len, curr_task_id):
        # task_ent_output  # :截至当前任务的 0:offset_e
        seq_len_mask = sequence_mask(seq_len)  # b,l
        ofs, ofe = self.task_offset_lst[curr_task_id]  # 当前任务的offset是
        if curr_task_id == 0:  # 第一个任务
            ce_mask = seq_len_mask  # ce_mask [b,l] 哪些是要计算ce_loss的 第一个任务所有都要。
            predict = ent_output[:, :, :ofe]
            ce_loss = self.calc_ce_loss(ce_target, predict, seq_len_mask, ce_mask)
            kl_loss = torch.tensor(0.)

        else:  # 后续任务
            ce_mask = torch.logical_and(ce_target >= ofs, ce_target < ofe).float()  # 之后的任务当前token属于新任务的ent时要
            predict = ent_output[:, :, :ofe]
            ce_loss = self.calc_ce_loss(ce_target, predict, seq_len_mask, ce_mask)

            kl_mask = 1. - ce_mask
            # kl_predict = ent_output[:, :, :ofs]
            kl_predict = ent_output[:, :, :ofe]  # 用 小值来补新标签对应的B和I
            # print(kl_target.shape)
            kl_target = torch.nn.functional.pad(kl_target, (0, ofe - ofs), mode='constant', value=-1e8)
            # print(kl_target.shape)
            # print(kl_predict.shape)
            # ipdb.set_trace()
            kl_loss = self.calc_kl_loss(kl_predict, kl_target, seq_len_mask, kl_mask, ofs, ofe)

        return ce_loss, kl_loss

    def runloss(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]

        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']

        ent_output = self.encode(inputs_dct)

        # 过滤其他任务的ent
        ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        curr_task_batch_tag_ids = batch_tag_ids.masked_fill(torch.logical_or(batch_tag_ids >= ofe, batch_tag_ids < ofs), 0.)  # [b,l]
        # print(inputs_dct['batch_ner_exm'][0])
        # print(batch_tag_ids[0])
        # print(curr_task_batch_tag_ids[0])
        # ipdb.set_trace()
        ce_loss, kl_loss = self.calc_loss(curr_task_batch_tag_ids, batch_distilled_task_ent_output, ent_output, seq_len, task_id)

        self.ce_loss = ce_loss
        # self.ce_loss = ce_loss.mean()
        self.kl_loss = kl_loss
        # self.kl_loss = kl_loss.mean()

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )

    def run_loss_non_cl(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]

        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']

        ent_output = self.encode(inputs_dct)

        # 过滤后面任务的ent
        ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        curr_task_batch_tag_ids = batch_tag_ids.masked_fill(batch_tag_ids >= ofe, 0.)  # [b,l]
        seq_len_mask = sequence_mask(seq_len)  # b,l

        ce_mask = seq_len_mask  # ce_mask [b,l] 哪些是要计算ce_loss的 第一个任务所有都要。
        predict = ent_output[:, :, :ofe]
        self.ce_loss = self.calc_ce_loss(curr_task_batch_tag_ids, predict, seq_len_mask, ce_mask)

        self.kl_loss = 0.

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )


class BaselineAdd(NerModel):
    def __init__(self, args, loader):
        super(BaselineAdd, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.grad_clip = None
        if args.corpus == 'onto' or args.corpus == 'onto_fs':
            self.grad_clip = 1.0
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        self.use_bert = args.pretrain_mode == 'fine_tuning'
        if self.use_bert:
            self.dropout_layer = nn.Dropout(p=args.enc_dropout)
            self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
            self.bert_layer = BertModel.from_pretrained(args.bert_model_dir)
            encoder_dim = self.bert_conf.hidden_size
            # self.lr = 5e-5
            # self.lr = 1e-4
            self.lr = self.args.bert_lr  # 5e-5
        else:
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )
            encoder_dim = 1024
            self.lr = self.args.lr  # 1e-3

        num_total_ents = sum(self.num_ents_per_task)

        # Add NER
        self.ent_layer = torch.nn.Linear(encoder_dim, self.num_tasks + num_total_ents * 2)  # B I for each ent, O for each task AddNER
        # print('total_params:', sum(p.numel() for p in self.parameters()))
        # print(*[n for n, p in self.named_parameters()], sep='\n')

        self.task_offset_lst = []  # the dim position in ent_layer
        offset_s = 0
        offset_e = None
        for num_ents in self.num_ents_per_task:
            if not self.task_offset_lst:
                offset_e = 1 + num_ents * 2
                self.task_offset_lst.append([offset_s, offset_e])
            else:
                offset_s = offset_e
                offset_e = offset_e + (1 + num_ents * 2)
                self.task_offset_lst.append([offset_s, offset_e])

        self.offset_split = [e[0] for e in self.task_offset_lst][1:]  # 用以np.split
        # np.split(tag_logtis, offset_split, axis=-1) split each task of ent_layer [O,B,I][O,B,I]...

        self.taskid2tagid_range = {}  # the BIO-tag id in tag2id dict, use to process the original input label.
        for task_id in range(self.num_tasks):
            start_ent = self.loader.entity_task_lst[task_id][0]
            end_ent = self.loader.entity_task_lst[task_id][-1]
            self.taskid2tagid_range[task_id] = [self.loader.datareader.tag2id[f'B-{start_ent}'],
                                                self.loader.datareader.tag2id[f'I-{end_ent}']
                                                ]
        print('task_offset_lst (task offset in ent_layer dim)', self.task_offset_lst)  # e.g. fewnerd [[0, 15], [15, 32], [32, 53], [53, 78], [78, 97], [97, 114], [114, 127], [127, 140]]
        print('offset_split (num_dim in ent_layer per task)', self.offset_split)  # e.g. fewnerd   # [15, 32, 53, 78, 97, 114, 127]
        print('taskid2tagid_range', self.taskid2tagid_range)  # e.g. fewnerd {0: [1, 14], 1: [15, 30], 2: [31, 50], 3: [51, 74], 4: [75, 92], 5: [93, 108], 6: [109, 120], 7: [121, 132]}

        # e.g. onto
        # task_offset_lst [[0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [15, 18]]
        # offset_split [3, 6, 9, 12, 15]
        # taskid2tagid_range {0: [1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8], 4: [9, 10], 5: [11, 12]}

        # each task dummy id2tag
        self.tid_id2tag_lst = []
        for tid, ents in enumerate(loader.entity_task_lst):
            id2tag = {0: 'O'}
            for ent in ents:
                id2tag[len(id2tag)] = f'B-{ent}'
                id2tag[len(id2tag)] = f'I-{ent}'
            self.tid_id2tag_lst.append(id2tag)

        self.ce_loss_layer = nn.CrossEntropyLoss(reduction='none')
        count_params(self)

        if self.use_bert:
            fast_lr = self.args.lr  # 1e-3
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            entlayer_p_weight = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'weight' in n]
            entlayer_p_bias = [p for n, p in self.named_parameters() if 'ent_layer' in n and 'bias' in n]
            p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and any(nd in n for nd in no_decay)]
            p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'ent_layer' not in n and not any(nd in n for nd in no_decay)]
            self.grouped_params = [
                {'params': entlayer_p_weight, 'lr': fast_lr},
                {'params': entlayer_p_bias, 'weight_decay': 0.0, 'lr': fast_lr},
                {'params': p2, 'weight_decay': 0.0},
                {'params': p3},
            ]
            if p1:  # using task emb
                self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': fast_lr}] + self.grouped_params

            check_param_groups(self, self.grouped_params)

        self.init_opt()
        if self.use_schedual:
            self.init_lrs()

    def encode(self, inputs_dct):
        batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
        seq_len = inputs_dct['ori_seq_len']

        if self.use_bert:
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)

            bert_out = self.dropout_layer(bert_out)  # don't forget
            encode_output = bert_out

        else:  # bilstm
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            encode_output = rnn_output_x

        # ffn
        encode_output = self.ent_layer(encode_output)  # [b,l,t]
        self.batch_tag_tensor = encode_output
        return encode_output

    def decode(self, ent_output, seq_len, task_id):
        offset_s, offset_e = self.task_offset_lst[task_id]

        mask = sequence_mask(seq_len, dtype=torch.uint8)
        # decode_ids = self.crf_layer.decode(ent_output, mask)

        ent_output_prob = ent_output[:, :, :offset_e].softmax(-1)

        id2tag = self.loader.datareader.id2tag
        curr_task_id2tag = {i: t for i, t in id2tag.items() if i < offset_e}
        curr_task_tag2id = {t: i for i, t in curr_task_id2tag.items()}
        curr_task_num_tags = len(curr_task_id2tag)
        trans_mask = get_BIO_transitions_mask(curr_task_tag2id, curr_task_id2tag)
        trans_mask = torch.tensor(1 - trans_mask).to(ent_output_prob.device)
        num_valid_for_prev_tag = trans_mask.sum(-1)
        trans_mask = trans_mask / num_valid_for_prev_tag
        start_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)
        end_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)

        decode_ids = viterbi_decode(ent_output_prob, mask, start_transitions, end_transitions, trans_mask)
        return decode_ids

    def decode_one_task(self, ent_output, seq_len, task_id):
        offset_s, offset_e = self.task_offset_lst[task_id]

        mask = sequence_mask(seq_len, dtype=torch.uint8)
        # decode_ids = self.crf_layer.decode(ent_output, mask)

        ent_output_prob = ent_output[:, :, offset_s:offset_e].softmax(-1)

        curr_task_id2tag = self.tid_id2tag_lst[task_id]
        curr_task_tag2id = {t: i for i, t in curr_task_id2tag.items()}
        curr_task_num_tags = len(curr_task_id2tag)
        trans_mask = get_BIO_transitions_mask(curr_task_tag2id, curr_task_id2tag)
        trans_mask = torch.tensor(1 - trans_mask).to(ent_output_prob.device)
        num_valid_for_prev_tag = trans_mask.sum(-1)
        trans_mask = trans_mask / num_valid_for_prev_tag
        start_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)
        end_transitions = torch.full([curr_task_num_tags], 1 / curr_task_num_tags).to(ent_output_prob.device)

        decode_ids = viterbi_decode(ent_output_prob, mask, start_transitions, end_transitions, trans_mask)
        return decode_ids

    def calc_ce_loss(self, target, predict, seq_len_mask, ce_mask):
        # target [b,l]  predict [b,l,ent]
        ce_loss = self.ce_loss_layer(predict.transpose(1, 2), target)  # [b,ent,l] [b,l] -> [b,l]
        ce_loss = ce_loss * seq_len_mask
        ce_loss = ce_loss * ce_mask
        # return span_loss。mean()  # [b,l]
        return ce_loss  # [b,l]

    def calc_kl_loss(self, predict, target, seq_len_mask, kl_mask):
        # kl_loss = torch.nn.functional.kl_div(log_kl_pred.double(), target.double(), reduction='none')  # [num_spans, ent, 2]
        # target = target / 2.  # temperature
        target_prob = target.softmax(dim=-1)
        # curr_ent_dim = ofe - ofs
        # pad_tenosr = torch.zeros_like(target_prob)[:, :, :curr_ent_dim]
        # target_prob = torch.cat([target_prob, pad_tenosr], dim=-1)

        kl_loss = torch.nn.functional.kl_div(predict.softmax(dim=-1).log(), target_prob, reduction='none')  # [b,l,dim]
        kl_loss = kl_loss.sum(-1)  # [b,l]
        kl_loss = kl_loss * seq_len_mask
        kl_loss = kl_loss * kl_mask

        return kl_loss  # [b,l]

    def runloss(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]
        seq_len_mask = sequence_mask(seq_len)

        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']

        ent_output = self.encode(inputs_dct)

        # 过滤其他任务的ent
        ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        tagid_s, tagid_e = self.taskid2tagid_range[task_id]

        need_to_change_mask = torch.logical_and(batch_tag_ids >= tagid_s, batch_tag_ids <= tagid_e)

        curr_task_batch_tag_ids = batch_tag_ids + need_to_change_mask.int() * (-ofs + task_id)

        curr_task_batch_tag_ids = curr_task_batch_tag_ids.masked_fill(torch.logical_not(need_to_change_mask), 0.)  # [b,l]
        self.ce_loss = self.calc_ce_loss(curr_task_batch_tag_ids, ent_output[:, :, ofs:ofe], seq_len_mask, seq_len_mask)
        self.ce_loss = self.ce_loss.mean()
        # self.ce_loss = self.ce_loss.mean(-1)
        # self.ce_loss = self.ce_loss.mean(-1)

        if task_id > 0:
            all_task_kl_loss = []
            for tid in range(task_id):
                ofs, ofe = self.task_offset_lst[tid]
                task_kl_loss = self.calc_kl_loss(ent_output[:, :, ofs:ofe],
                                                 batch_distilled_task_ent_output[:, :, ofs:ofe],
                                                 seq_len_mask, kl_mask=seq_len_mask)  # [b,l]
                task_kl_loss = torch.mean(task_kl_loss)
                all_task_kl_loss.append(task_kl_loss)
            self.kl_loss = sum(all_task_kl_loss) / len(all_task_kl_loss)
        else:
            self.kl_loss = 0

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )

    def run_loss_non_cl(self, inputs_dct, task_id):
        seq_len = inputs_dct['ori_seq_len']
        batch_tag_ids = inputs_dct['batch_tag_ids']  # [b,l]
        seq_len_mask = sequence_mask(seq_len)
        batch_distilled_task_ent_output = inputs_dct['batch_distilled_task_ent_output']
        ent_output = self.encode(inputs_dct)

        ce_loss_lst = []
        for tid in range(task_id + 1):
            # 过滤其他任务的ent
            ofs, ofe = self.task_offset_lst[tid]  # 当前任务的offset是
            tagid_s, tagid_e = self.taskid2tagid_range[tid]

            need_to_change_mask = torch.logical_and(batch_tag_ids >= tagid_s, batch_tag_ids <= tagid_e)

            curr_task_batch_tag_ids = batch_tag_ids + need_to_change_mask.int() * (-ofs + tid)

            curr_task_batch_tag_ids = curr_task_batch_tag_ids.masked_fill(torch.logical_not(need_to_change_mask), 0.)  # [b,l]
            ce_loss = self.calc_ce_loss(curr_task_batch_tag_ids, ent_output[:, :, ofs:ofe], seq_len_mask, seq_len_mask)
            ce_loss = ce_loss.mean()
            # ce_loss = ce_loss.mean(-1)
            # ce_loss = ce_loss.mean(-1)
            ce_loss_lst.append(ce_loss)

        self.ce_loss = sum(ce_loss_lst) / len(ce_loss_lst)

        self.kl_loss = 0.

        self.total_loss = self.ce_loss + self.kl_loss

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()

        if self.use_schedual:
            self.lrs.step()

        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.kl_loss),
                )


class SpanKL(NerModel):
    def __init__(self, args, loader):
        super(SpanKL, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.taskid2offset = loader.tid2offset  # 每个任务不同个ent
        self.hidden_size_per_ent = 50
        self.grad_clip = None
        if args.corpus == 'onto' or args.corpus == 'onto_fs':
            self.grad_clip = 1.0
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        self.gumbel_generator = torch.Generator(device=args.device)
        self.gumbel_generator.manual_seed(self.args.seed)
        # self.ent_size = self.num_ent = self.loader.ent_dim  # total ent dim across all task
        self.ep = None
        self.use_slr = False  # not used in the published paper

        self.use_bert = args.pretrain_mode == 'fine_tuning'
        if self.use_bert:
            self.dropout_layer = nn.Dropout(p=args.enc_dropout)
            self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
            self.bert_layer = BertModel.from_pretrained(args.bert_model_dir)
            encoder_dim = self.bert_conf.hidden_size
            # self.lr = 5e-5
            # self.lr = 1e-4
            self.lr = self.args.bert_lr

            # self.extra_dense = torch.nn.Linear(encoder_dim, encoder_dim)
        else:
            self.bilstm_layer = nn.LSTM(
                input_size=1024,
                hidden_size=512,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=0
            )
            encoder_dim = 1024
            self.lr = self.args.lr  # 1e-3

        if self.use_slr:
            self.slr_layer = torch.nn.Linear(encoder_dim, 50 * 2)

        if self.args.use_task_embed:
            # self.task_embed = torch.nn.Embedding(self.num_tasks, 1024)
            self.task_embed = torch.nn.Parameter(torch.Tensor(self.num_tasks, encoder_dim))
            # torch.nn.init.uniform_(self.task_embed.data, -1., 1.)
            torch.nn.init.normal_(self.task_embed.data, mean=0., std=1.)
            # torch.nn.init.normal_(self.task_embed.data, mean=0., std=0.25)  # 不行 会导致更少的门开
            # torch.nn.init.constant_(self.task_embed.data, 0)
            # self.task_embed.data = self.task_embed.data.abs()  # 门全部先开着

            self.gate_tensor_lst = [None] * self.num_tasks  # 用来存放当前前向传播时经过gumble_softmax采样得到的0-1binary vector

        # task dense layer
        self.output_dim_per_task_lst = [2 * self.hidden_size_per_ent * num_ents for num_ents in self.num_ents_per_task]
        # self.task_layers = [nn.Linear(encoder_dim, output_dim_per_task) for output_dim_per_task in self.output_dim_per_task_lst]  # one task one independent layer
        # 这样才能cuda()生效
        self.task_layers = nn.ModuleList([torch.nn.Linear(encoder_dim, output_dim_per_task) for output_dim_per_task in self.output_dim_per_task_lst])  # one task one independent layer
        for layer in self.task_layers:
            torch.nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)

        self.bce_loss_layer = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss_layer = torch.nn.MSELoss(reduction='none')
        count_params(self)

        # print(*[n for n, p in self.named_parameters()], sep='\n')
        # ipdb.set_trace()

        if self.use_bert:
            """1"""
            no_decay = ['bias', 'LayerNorm.weight']
            p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            spanlayer_p_weight = [p for n, p in self.named_parameters() if 'task_layers' in n and 'weight' in n]
            spanlayer_p_bias = [p for n, p in self.named_parameters() if 'task_layers' in n and 'bias' in n]
            p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and any(nd in n for nd in no_decay)]
            p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and not any(nd in n for nd in no_decay)]
            self.grouped_params = [
                {'params': spanlayer_p_weight, 'lr': 1e-3},
                {'params': spanlayer_p_bias, 'weight_decay': 0.0, 'lr': 1e-3},
                {'params': p2, 'weight_decay': 0.0},
                {'params': p3},
            ]
            if p1:  # using task emb
                self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': 1e-3}] + self.grouped_params
            """2"""
            # no_decay = ['bias', 'LayerNorm.weight']
            # p1 = [p for n, p in self.named_parameters() if n == 'task_embed']
            # spanlayer_p_weight = [p for n, p in self.named_parameters() if 'task_layers' in n and 'weight' in n]
            # spanlayer_p_bias = [p for n, p in self.named_parameters() if 'task_layers' in n and 'bias' in n]
            # extralayer_w = [p for n, p in self.named_parameters() if 'extra_dense' in n and 'weight' in n]
            # extralayer_b = [p for n, p in self.named_parameters() if 'extra_dense' in n and 'bias' in n]
            # p2 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and 'extra_dense' not in n and any(nd in n for nd in no_decay)]
            # p3 = [p for n, p in self.named_parameters() if n != 'task_embed' and 'task_layers' not in n and 'extra_dense' not in n and not any(nd in n for nd in no_decay)]
            # self.grouped_params = [
            #     {'params': spanlayer_p_weight, 'lr': 1e-3},
            #     {'params': spanlayer_p_bias, 'weight_decay': 0.0, 'lr': 1e-3},
            #     {'params': extralayer_w, 'lr': 1e-3},
            #     {'params': extralayer_b, 'weight_decay': 0.0, 'lr': 1e-3},
            #     {'params': p2, 'weight_decay': 0.0},
            #     {'params': p3},
            # ]
            # if p1:  # using task emb
            #     self.grouped_params = [{'params': p1, 'weight_decay': 0.0, 'lr': 1e-3}]+ self.grouped_params
            """ """
            check_param_groups(self, self.grouped_params)

        self.init_opt()
        if self.use_schedual:
            self.init_lrs()

        # ipdb.set_trace()

    def encoder_forward(self, inputs_dct):
        if self.use_bert:
            seq_len = inputs_dct['seq_len']  # seq_len [bat] 原始句子分词后的子token个数（不包含cls和sep）（≥原始token数）
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state  # (32,49,768)
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch  # 去除[CLS]和[SEP]后变为(32,47,768)

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)   # (32,40,768)，这个40表示这个batch的32个样例中，最长的词个数（没有分成子词），即batch_exm中的char_lst长度

            bert_out = self.dropout_layer(bert_out)  # don't forget

            # bert_out = torch.tanh(bert_out)
            # bert_out = self.extra_dense(bert_out)
            return bert_out
        else:
            batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
            seq_len = inputs_dct['ori_seq_len']
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            return rnn_output_x

    def task_layer_forward(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, deterministic=False):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1}
                    if deterministic:  # when are not training, which unable grad back propagation
                        gate = (gate_logit >= 0.).float()  # deterministic output
                    else:  # when training, which enable grad back propagation
                        # bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
                        # gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def task_layer_forward1(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, deterministic_tasks=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1}
                    if task_id in deterministic_tasks:  # when are not training, which unable grad back propagation
                        gate = (gate_logit >= 0.).float()  # deterministic output
                    else:  # when training, which enable grad back propagation
                        # bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
                        # gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def task_layer_forward2(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, gumbel_tasks=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1} with random sample. logits is distribution
                    if task_id in gumbel_tasks:  # when train specific task, which enable grad back propagation
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                        # gate = torch.ones_like(gate_logit)
                        # self.gate_tensor_lst[task_id] = gate_logit
                    else:
                        # 这种方式不会传梯度到gate_logit也就是task_embed去。
                        gate = (gate_logit >= 0.).float()  # deterministic output. when is not training, which unable grad back propagation
                        # if task_id == 0:
                        #     gate = (gate_logit > 0.).float()
                        # else:
                        #     gate = torch.ones_like(gate_logit)
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]

        if self.use_slr:
            slr_output = self.slr_layer(encoder_output)  # [b,l,2h]
            link_start_hidden, link_end_hidden = torch.chunk(slr_output, 2, dim=-1)
            link_scores = calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            pooling_type = 'softmin'
            logsumexp_temp = 0.3
            self.refined_scores = calc_refined_mat_tensor(link_scores, pooling_type=pooling_type, temp=logsumexp_temp)  # b,l,l,1

        return output_per_task_lst

    def task_layer_forward3(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, gumbel_tasks=None, ep=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1} with random sample. logits is distribution
                    if task_id in gumbel_tasks:  # when train specific task, which enable grad back propagation
                        # if ep == 0:
                        #     gate = torch.ones_like(gate_logit)
                        #     self.gate_tensor_lst[task_id] = gate_logit
                        # elif ep == 1:
                        #     gate = gumbel_sigmoid(gate_logit, hard=False, use_gumbel=False, tau=1/3, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # elif ep > 1:
                        #     gate = gumbel_sigmoid(gate_logit, hard=True, use_gumbel=True, tau=1/3, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # if ep <= 6:
                        #     gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # else:
                        #     gate = (gate_logit > 0.).float()
                        #     self.gate_tensor_lst[task_id] = gate
                        if ep is not None:
                            gate = (gate_logit >= 0.).float()
                            self.gate_tensor_lst[task_id] = gate

                    else:
                        # 这种方式不会传梯度到gate_logit也就是task_embed去。
                        gate = (gate_logit > 0.).float()  # deterministic output. when is not training, which unable grad back propagation
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def span_matrix_forward(self, output, seq_len):
        bsz, length = output.shape[:2]
        # decode the output from output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)
        ent_hid_lst = torch.split(output, self.output_dim_per_task_lst, dim=-1)
        start_hid_lst = [torch.chunk(t, 2, dim=-1)[0].reshape([bsz, length, self.hidden_size_per_ent, -1]) for t in ent_hid_lst]  # list of [b,l,hid,ent](for one task)
        end_hid_lst = [torch.chunk(t, 2, dim=-1)[1].reshape([bsz, length, self.hidden_size_per_ent, -1]) for t in ent_hid_lst]  # list of [b,l,hid,ent](for one task)
        start_hidden = torch.cat(start_hid_lst, dim=-1)  # b,l,h,e
        end_hidden = torch.cat(end_hid_lst, dim=-1)  # # b,l,h,e
        start_hidden = start_hidden.permute(0, 3, 1, 2)  # b,e,l,h
        end_hidden = end_hidden.permute(0, 3, 1, 2)  # b,e,l,h

        total_ent_size = start_hidden.shape[1]

        attention_scores = torch.matmul(start_hidden, end_hidden.transpose(-1, -2))  # [bat,num_ent,len,hid] * [bat,num_ent,hid,len] = [bat,num_ent,len,len]
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_ent)
        span_ner_mat_tensor = attention_scores.permute(0, 2, 3, 1)  # b,l,l,e
        if self.use_slr:
            span_ner_mat_tensor = span_ner_mat_tensor + self.refined_scores

        self.batch_span_tensor = span_ner_mat_tensor
        # 构造下三角mask 去除了pad和下三角区域
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask pad为0
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 下三角0 上三角和对角线1
        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask[..., None])  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, total_ent_size)  # [*,ent]
        return span_ner_pred_lst

    def compute_offsets(self, task_id, mode='train'):
        ofs_s, ofs_e = self.taskid2offset[task_id]
        if mode == 'train':
            pass
        if mode == 'test':
            ofs_s = 0  # 从累计最开始的task算起
        return int(ofs_s), int(ofs_e)

    def calc_loss(self, batch_span_pred, batch_span_tgt):
        # pred  batch_span_pred [num_spans,ent]
        # label batch_span_tgt  [num_spans,ent]
        span_loss = self.bce_loss_layer(batch_span_pred, batch_span_tgt)  # [*,ent] [*,ent](target已是onehot) -> [*,ent]
        # ipdb.set_trace()
        # span_loss = torch.sum(span_loss, -1)  # [*] over ent
        span_loss = torch.mean(span_loss, -1)  # [*] over ent
        # span_loss = torch.mean(span_loss)  # 这样loss是0.00x 太小优化不了
        span_loss = torch.sum(span_loss, -1)  # [] over num_spans
        # span_loss = torch.mean(span_loss, -1)  # [] over num_spans
        return span_loss  # []

    def calc_f1(self, batch_span_pred, batch_span_tgt):
        # pred batch_span_pred [*,ent], label batch_span_tgt [*,ent]
        batch_span_pred = (batch_span_pred > 0).int()  # [*,ent] logits before sigmoid
        # calc f1
        num_pred = torch.sum(batch_span_pred)
        num_gold = torch.sum(batch_span_tgt)
        tp = torch.sum(batch_span_tgt * batch_span_pred)
        f1 = torch.tensor(1.) if num_gold == num_pred == 0 else 2 * tp / (num_gold + num_pred + 1e-12)
        return f1.item(), (num_gold.item(), num_pred.item(), tp.item())

    def calc_kl_loss(self, batch_target_lst_distilled, batch_predict_lst_need_distill, task_id):
        ofs_s, ofs_e = self.compute_offsets(task_id)
        bsz = len(batch_target_lst_distilled)  # 32
        # kl_losses = []
        batch_kl_loss = 0.
        for bdx in range(bsz):  # 只蒸馏当前的样本 忽略记忆库的
            pred_need_distill = batch_predict_lst_need_distill[bdx][:, :ofs_s]  # 取当前任务之前的所有任务上的预测结果（这个结果完全是蒸馏模型的输出）
            kl_pred = torch.stack([pred_need_distill, -pred_need_distill], dim=-1)  # [num_spans, ent, 2]
            log_kl_pred = torch.nn.functional.logsigmoid(kl_pred)

            kl_tgt = batch_target_lst_distilled[bdx][:, :ofs_s]  # 蒸馏的结果（教师模型的评估结果）

            # kl_tgt = torch.stack([kl_tgt, 1. - kl_tgt], dim=-1)  # [num_spans, ent, 2]  # kl_tgt为prob
            # kl_tgt_logits = -torch.log(1 / (tgt_prob + 1e-8) - 1 + 1e-8)  # inverse of sigmoid
            # kl_loss = torch.nn.functional.kl_div(log_kl_pred, kl_tgt, reduction='none')  # [num_spans, ent, 2]

            kl_tgt = kl_tgt / 1.  # temp=1
            kl_tgt_logit = torch.stack([kl_tgt, -kl_tgt], dim=-1)  # [num_spans, ent, 2]  # kl_tgt为logits
            log_kl_tgt = torch.nn.functional.logsigmoid(kl_tgt_logit)
            kl_loss = torch.nn.functional.kl_div(log_kl_pred, log_kl_tgt, reduction='none', log_target=True)  # [num_spans, ent, 2]

            kl_loss = torch.sum(kl_loss, -1)  # kl definition
            kl_loss = torch.mean(kl_loss, -1)  # over ent
            kl_loss = torch.sum(kl_loss, -1)  # over spans
            # kl_loss = torch.mean(kl_loss, -1)  # over spans

            # kl_losses.append(kl_loss)
            # self.kl_loss = sum(kl_losses) / len(kl_losses)
            batch_kl_loss += kl_loss

        # return torch.abs(batch_kl_loss / bsz)
        return batch_kl_loss / bsz
        # return torch.relu(batch_kl_loss / bsz)

    def calc_kl_loss_mse(self, batch_target_lst_distilled, batch_predict_lst_need_distill, task_id):
        ofs_s, ofs_e = self.compute_offsets(task_id)
        bsz = len(batch_target_lst_distilled)
        # kl_losses = []
        batch_kl_loss = 0.
        for bdx in range(bsz):  # 只蒸馏当前的样本 忽略记忆库的
            kl_tgt = batch_target_lst_distilled[bdx][:, :ofs_s]  # prob

            pred_need_distill = batch_predict_lst_need_distill[bdx][:, :ofs_s]
            mse_loss = self.mse_loss_layer(pred_need_distill.sigmoid(), kl_tgt)  # [num_spans, ent]

            mse_loss = torch.mean(mse_loss, -1)  # over ent
            mse_loss = torch.sum(mse_loss, -1)  # over spans
            # mse_loss = torch.mean(mse_loss, -1)  # over spans

            # kl_losses.append(kl_loss)
            # self.kl_loss = sum(kl_losses) / len(kl_losses)
            batch_kl_loss += mse_loss

        return batch_kl_loss / bsz

    def take_loss(self, task_id, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e = self.compute_offsets(task_id)
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头  只计算当前任务的loss
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def take_multitask_loss(self, batch_task_id, batch_predict_lst, batch_target_lst, f1_meaner=None):
        """multiple task of example"""  # batch_predict_lst,batch_target_lst: list of [num_spans, ent]
        losses = []
        for bdx, task_id in enumerate(batch_task_id):  # 以样本为单位计算loss 该batch有多个任务
            ofs_s, ofs_e = self.compute_offsets(task_id)
            span_loss = self.calc_loss(batch_predict_lst[bdx][:, ofs_s:ofs_e], batch_target_lst[bdx][:, ofs_s:ofs_e])  # 只计算对应task的头
            f1, f1_detail = self.calc_f1(batch_predict_lst[bdx][:, ofs_s:ofs_e], batch_target_lst[bdx][:, ofs_s:ofs_e])
            losses.append(span_loss)
            if f1_meaner is not None:
                f1_meaner.add(*f1_detail)
        loss = sum(losses) / len(losses)
        return loss

    def eval_forward(self, inputs_dct, task_id, mode='train'):
        # 用于eval
        seq_len = inputs_dct['ori_seq_len']
        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=[])
        batch_span_pred = self.span_matrix_forward(task_layer_output, seq_len)  # [bsz*num_spans, ent] (9918,6)
        batch_predict_lst = torch.split(batch_span_pred, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        ofs_s, ofs_e = self.compute_offsets(task_id, mode=mode)  # make sure we predict classes within the current task

        f1, detail_f1, span_loss, kl_loss = None, None, None, None
        if 'batch_span_tgt' in inputs_dct:  # if label had passed into
            batch_span_tgt = inputs_dct['batch_span_tgt']  # [bsz*num_spans, ent]
            f1, detail_f1 = self.calc_f1(batch_span_pred[:, ofs_s:ofs_e], batch_span_tgt[:, ofs_s:ofs_e])

            span_loss = self.take_loss(task_id, batch_span_pred, batch_span_tgt, bsz=len(seq_len))  # 默认是curr(train)
        if self.args.use_distill and task_id > 0 and inputs_dct.get('batch_span_tgt_lst_distilled', None):
            kl_loss = self.calc_kl_loss(inputs_dct['batch_span_tgt_lst_distilled'], batch_predict_lst, task_id)  # 默认是so_far-curr

        return batch_span_pred[:, ofs_s:ofs_e], f1, detail_f1, span_loss, kl_loss

    def forward(self, *args, **kwargs):   # 评估时调用此方法
        return self.eval_forward(*args, **kwargs)

    def observe(self, inputs_dct, task_id, f1_meaner, ep=None):   # 训练时调用此方法
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']   # [all_num_spans, ent]
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)   # (32,40,768)，其中40表示的是句子中词个数（不是子词个数）

        if ep is not None:
            task_layer_output = self.task_layer_forward3(encoder_output,
                                                         use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                         gumbel_tasks=[task_id],
                                                         ep=ep)
        else:
            task_layer_output = self.task_layer_forward2(encoder_output,  # (32,40,600)
                                                         use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                         gumbel_tasks=[task_id],
                                                         )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length) # (6775,6) # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        # self.span_loss1 = self.take_multitask_loss([task_id] * bsz, batch_predict_lst, batch_target_lst, f1_meaner)
        self.span_loss = self.take_loss(task_id, batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # loss按batch平均才能跟kl对齐
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_distill and task_id > 0:
            batch_predict_lst_need_distill = batch_predict_lst
            self.kl_loss = self.calc_kl_loss(batch_target_lst_distilled, batch_predict_lst_need_distill, task_id)

            self.total_loss += self.kl_loss
            # self.total_loss += self.kl_loss / (self.kl_loss.detach() + 1e-7)

            # kl_coe = self.span_loss.detach() / (self.kl_loss.detach() + 1e-7)
            # self.kl_loss *= kl_coe
            # self.total_loss += self.kl_loss

        if self.args.use_task_embed:
            if self.args.use_gumbel_softmax:
                # curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob
            else:
                curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
            # print(curr_task_gate.sum(-1))
            # 这里norm_loss 如果值为1则有梯度1/1024, 如果值为0则有梯度为0,相当于不优化为0的门。即prob=0.49的门有可能越过成0.51.
            self.sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
            # print('\ncurr_task_gate', curr_task_gate.sum(-1))
            # print('sparse_loss', self.sparse_loss)
            # task_embed_binary = (self.task_embed[task_id] > 0).float()
            # print('task_embed', task_embed_binary.sum(-1))
            # print('sparse_loss\n', torch.norm(task_embed_binary, p=1, dim=-1) / task_embed_binary.shape[0])

            # self.sparse_loss *= 0.5
            # if task_id > 0:
            #     self.sparse_loss *= 2
            # self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

            sparse_coe = self.total_loss.detach() / (self.sparse_loss.detach())
            # self.sparse_loss *= sparse_coe
            # self.total_loss += self.sparse_loss * sparse_coe
            # if self.ep and self.ep > 0:
            #     self.total_loss += self.sparse_loss * sparse_coe * 0.5
            self.total_loss += self.sparse_loss * sparse_coe * 0.5

            # self-entropy loss
            # task_prob = torch.sigmoid(self.task_embed[task_id])
            # entropy_loss = - task_prob * torch.log(task_prob) - (1. - task_prob) * torch.log(1. - task_prob)
            # self.entropy_loss = torch.mean(entropy_loss, -1)
            # entropy_coe = self.total_loss.detach() / (self.entropy_loss.detach())

            # self.total_loss += self.entropy_loss * entropy_coe * 0.5

            # 0602
            # self.sparse_loss *= self.total_loss.detach() / (self.sparse_loss.detach() + 1e-7)
            # self.total_loss += self.sparse_loss

        # if self.args.use_task_embed:
        #     if self.args.use_gumbel_softmax:
        #         s_l_lst = []
        #         for tid in range(task_id+1):
        #             curr_task_gate = self.gate_tensor_lst[tid]  # [emb_dim] 0-1 binary
        #             _sparse_loss = torch.norm(curr_task_gate, p=1, dim=0) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
        #             s_l_lst.append(_sparse_loss)
        #         self.sparse_loss = sum(s_l_lst) / len(s_l_lst)
        #     else:
        #         curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob
        #     self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()
        # if task_id > 0:
        #     ipdb.set_trace()

        # ori_task_embed = copy.deepcopy(self.net.task_embed)
        # ipdb.set_trace()
        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)  # 梯度裁剪,目的是限制梯度的范数，以防止梯度爆炸的问题

        self.opt.step()
        # if ep is not None:
        #     if ep <= 6:
        #         self.lrs.step()
        #     else:
        #         pass
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )

    def take_alltask_loss(self, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        loss = self.calc_loss(batch_predict, batch_target)  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict, batch_target)
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def observe_all(self, inputs_dct, f1_meaner, ep=None):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.opt.zero_grad()

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=list(range(self.num_tasks)),
                                                     )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        self.span_loss = self.take_alltask_loss(batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # 这样loss不能按batch平均
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_task_embed:
            for task_id in range(self.num_tasks):
                if self.args.use_gumbel_softmax:
                    curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                else:
                    curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
                sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
                self.sparse_loss += sparse_loss
            self.sparse_loss /= self.num_tasks
            self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )

    def take_so_far_task_loss(self, task_id, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e = self.compute_offsets(task_id, mode='test')
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def observe_non_cl(self, inputs_dct, task_id, f1_meaner):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.opt.zero_grad()

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=list(range(self.num_tasks)),
                                                     )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        self.span_loss = self.take_so_far_task_loss(task_id, batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # 这样loss不能按batch平均
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_task_embed:
            for task_id in range(self.num_tasks):
                if self.args.use_gumbel_softmax:
                    curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                else:
                    curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
                sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
                self.sparse_loss += sparse_loss
            self.sparse_loss /= self.num_tasks
            self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )


class MLM(NerModel):
    def __init__(self, args, loader):
        super(MLM, self).__init__()
        self.args = args
        self.loader = loader
        self.num_tasks = loader.num_tasks
        self.num_ents_per_task = loader.num_ents_per_task
        self.taskid2offset = loader.tid2offset  # 每个任务不同个ent
        self.hidden_size_per_ent = 50
        self.grad_clip = None
        if args.corpus == 'onto' or args.corpus == 'onto_fs':
            self.grad_clip = 1.0
            # self.ori_label_token_map = {"I-ORG": ["National", "Corp", "News", "Inc", "Senate", "Court"],
            #                             "I-PERSON": ["John", "David", "Peter", "Michael", "Robert", "James"],
            #                             "I-GPE": ["US", "China", "United", "Beijing", "Israel", "Taiwan"],
            #                             "I-DATE": ["Year", "December", "August", "July", "1940", "March"],
            #                             "I-CARDINAL": ["Two", "four", "Three", "Hundred", "20", "8"],
            #                             "I-NORP": ["Chinese", "Israeli", "Palestinians", "American", "Japanese", "Palestinian"]
            #                             }
            # self.label_to_id = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-PERSON': 3, 'I-PERSON': 4, 'B-GPE': 5, 'I-GPE': 6, 'B-DATE': 7, 'I-DATE': 8, 'B-CARDINAL': 9, 'I-CARDINAL': 10, 'B-NORP': 11, 'I-NORP': 12}   # TODO 全局优化
        if args.corpus == 'conll':
            self.grad_clip = 1.0
            # self.ori_label_token_map = {"I-PER": ["Michael", "John", "David", "Thomas", "Martin", "Paul"],
            #                             "I-ORG": ["Corp", "Inc", "Commission", "Union", "Bank", "Party"],
            #                             "I-LOC": ["England", "Germany", "Australia", "France", "Russia", "Italy"],
            #                             "I-MISC": ["Palestinians", "Russian", "Chinese", "Dutch", "Russians", "English"]
            #                             }
            # self.label_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-MISC': 7, 'I-MISC': 8}   # TODO 全局优化
        if args.corpus == 'fewnerd':
            self.grad_clip = 5.0
        self.use_schedual = True
        # self.ent_size = self.num_ent = self.loader.ent_dim  # total ent dim across all task
        self.ep = None

        # rz+
        if args.pretrain_mode:
            # self.tokenizer = AutoTokenizer.from_pretrained(self.loader.bert_model_dir, use_fast=True, do_lower_case=False)
            self.tokenizer = self.loader.datareader.tokenizer
            self.use_bert = args.pretrain_mode
            if self.use_bert:
                self.dropout_layer = nn.Dropout(p=args.enc_dropout)
                # self.bert_conf = AutoConfig.from_pretrained(args.bert_model_dir)
                self.bert_conf = BertConfig.from_pretrained(args.bert_model_dir)
                # self.bert_layer = AutoModelForMaskedLM.from_pretrained(
                self.bert_layer = BertForMaskedLM.from_pretrained(
                                  args.bert_model_dir,
                                  from_tf=bool(".ckpt" in args.bert_model_dir),
                                  config=self.bert_conf,)
                # TODO 写优雅

                # def freeze_model(model):
                #     # 冻结除了 embedding 层、layernorm 层和 cls 层外的所有层
                #     for param in model.parameters():
                #         param.requires_grad = False
                #
                #     # 解冻 embedding 层、layernorm 层和 cls 层
                #     for name, param in model.named_parameters():
                #         if 'embeddings' in name or 'LayerNorm' in name or 'cls' in name:
                #             param.requires_grad = True
                #
                # freeze_model(self.bert_layer)

                # def freeze_model(model):
                #     for name, sub_module in model.named_modules():
                #         # if name == "extra_embeddings":  # 要冻结的模块名字
                #         # if "bert" in name:  # 要冻结的模块名字
                #         if "bert" in name or "LayerNorm" in name:  # 要冻结的模块名字
                #             for param_name, param in sub_module.named_parameters():
                #                 param.requires_grad = False
                #
                # freeze_model(self.bert_layer)

                # self.lr = 5e-5
                # self.lr = 1e-4
                self.lr = self.args.bert_lr
        else:
            pass

        # self.bce_loss_layer = torch.nn.BCEWithLogitsLoss(reduction='none')
        # self.mse_loss_layer = torch.nn.MSELoss(reduction='none')
        # count_params(self)

        if self.use_bert:   # TODO 这里是不是可以去掉，多余了
            no_decay = ['bias', 'LayerNorm.weight']
            self.grouped_params = [
                {
                    "params": [p for n, p in self.bert_layer.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in self.bert_layer.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]
            # check_param_groups(self, self.grouped_params)

        # self.init_opt()
        # if self.use_schedual:
        #     self.init_lrs()

    def encoder_forward(self, inputs_dct):
        if self.use_bert:
            seq_len = inputs_dct['seq_len']  # seq_len [bat]
            bert_outputs = self.bert_layer(input_ids=inputs_dct['input_ids'],
                                           token_type_ids=inputs_dct['bert_token_type_ids'],
                                           attention_mask=inputs_dct['bert_attention_mask'],
                                           output_hidden_states=True,
                                           )
            seq_len_lst = seq_len.tolist()
            bert_out = bert_outputs.last_hidden_state  # (32,49,768)
            # 去除bert_output[CLS]和[SEP]
            bert_out_lst = [t for t in bert_out]  # split along batch
            for i, t in enumerate(bert_out_lst):  # iter along batch
                # tensor [len, hid]
                bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            bert_out = torch.stack(bert_out_lst, 0)  # stack along batch  # 去除[CLS]和[SEP]后变为(32,47,768)

            batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
            if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
                bert_out_lst = [t for t in bert_out]
                for bdx, t in enumerate(bert_out_lst):
                    ori_2_tok = batch_ori_2_tok[bdx]
                    bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
                bert_out = torch.stack(bert_out_lst, 0)   # (32,40,768)，这个40表示这个batch的32个样例中，最长的词个数（没有分成子词），即batch_exm中的char_lst长度

            bert_out = self.dropout_layer(bert_out)  # don't forget

            # bert_out = torch.tanh(bert_out)
            # bert_out = self.extra_dense(bert_out)
            return bert_out
        else:
            batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
            seq_len = inputs_dct['ori_seq_len']
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            return rnn_output_x

    def pooling(self, sub, sup_mask, pool_type="mean"):
        sup = None
        if len(sub.shape) == len(sup_mask.shape):
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(0).repeat(1, sup_mask.shape[0], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup == -1e30] = 0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup == -1e30] = 0
        return sup

    def entlm_forward(self, input_ids, token_type_ids, attention_mask, seq_len, batch_ori_2_tok, batch_word_mask, batch_wo_pad_len):
        if self.use_bert:
            batch_entlm_input = {}
            batch_entlm_input['input_ids'] = input_ids
            batch_entlm_input['token_type_ids'] = token_type_ids
            batch_entlm_input['attention_mask'] = attention_mask
            batch_entlm_outputs_dict = self.bert_layer(**batch_entlm_input)
            # TODO 到底要不要dropout----->实验后发现，dropout后F1值略有提升（79.62升至80.06）
            '''
            dropout是将bert的输出（768维矩阵）中的某些值随机置为0，这么做是因为bert模型参数太多，
            相较于这种参数，我们平时的下游任务训练样本太少，因此容易导致过拟合现象，故在训练期间，
            将输出的一部分随机置为0，相当于变相减少了模型参数（因为随机的那部分模型参数不起作用了），
            但是在评估的过程中，不会使用dropout
            '''
            # bert_uncropped_outputs_lst = [batch_entlm_outputs_dict.logits[i, :length, :] for i, length in
            #                      enumerate(batch_wo_pad_len)]
            # batch_entlm_outputs_lst = []
            # for i, word_mask in enumerate(batch_word_mask):
            #     batch_entlm_outputs = self.pooling(bert_uncropped_outputs_lst[i], word_mask)
            #     batch_entlm_outputs_lst.append(batch_entlm_outputs)
            # max_seq_len = max(t.shape[1] for t in batch_entlm_outputs_lst)
            # batch_entlm_outputs = torch.zeros((len(batch_entlm_outputs_lst), max_seq_len, batch_entlm_outputs_lst[0].shape[-1]), device=bert_uncropped_outputs_lst[0].device)
            # for i, t in enumerate(batch_entlm_outputs_lst):
            #     batch_entlm_outputs[i, :t.shape[1], :] = t[0, :, :]

            # seq_len_lst = seq_len.tolist()
            # # 去除bert_output[CLS]和[SEP]
            # bert_out_lst = [t for t in batch_entlm_outputs_dict.logits]  # split along batch
            # for i, t in enumerate(bert_out_lst):  # iter along batch
            #     # tensor [len, hid]
            #     bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
            # bert_out = torch.stack(bert_out_lst, 0)  # stack along batch
            #
            # if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
            #     bert_out_lst = [t for t in bert_out]
            #     for bdx, t in enumerate(bert_out_lst):
            #         ori_2_tok = batch_ori_2_tok[bdx]
            #         bert_out_lst[bdx] = bert_out_lst[bdx][ori_2_tok]
            #     batch_entlm_outputs = torch.stack(bert_out_lst, 0)
            batch_entlm_outputs = batch_entlm_outputs_dict.logits
            return batch_entlm_outputs

        else:
            batch_input_pts = inputs_dct['batch_input_pts']  # [bsz, len, 1024]
            seq_len = inputs_dct['ori_seq_len']
            pack_embed = torch.nn.utils.rnn.pack_padded_sequence(batch_input_pts, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            pack_out, _ = self.bilstm_layer(pack_embed)
            rnn_output_x, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
            return rnn_output_x

    def task_layer_forward(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, deterministic=False):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1}
                    if deterministic:  # when are not training, which unable grad back propagation
                        gate = (gate_logit >= 0.).float()  # deterministic output
                    else:  # when training, which enable grad back propagation
                        # bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
                        # gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def task_layer_forward1(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, deterministic_tasks=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1}
                    if task_id in deterministic_tasks:  # when are not training, which unable grad back propagation
                        gate = (gate_logit >= 0.).float()  # deterministic output
                    else:  # when training, which enable grad back propagation
                        # bernoully_logits = torch.stack([gate_logit, -gate_logit], dim=0)
                        # gate = F.gumbel_softmax(bernoully_logits, tau=2/3, hard=True, dim=0)[0]
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def task_layer_forward2(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, gumbel_tasks=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1} with random sample. logits is distribution
                    if task_id in gumbel_tasks:  # when train specific task, which enable grad back propagation
                        gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        self.gate_tensor_lst[task_id] = gate
                        # gate = torch.ones_like(gate_logit)
                        # self.gate_tensor_lst[task_id] = gate_logit
                    else:
                        # 这种方式不会传梯度到gate_logit也就是task_embed去。
                        gate = (gate_logit >= 0.).float()  # deterministic output. when is not training, which unable grad back propagation
                        # if task_id == 0:
                        #     gate = (gate_logit > 0.).float()
                        # else:
                        #     gate = torch.ones_like(gate_logit)
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]

        if self.use_slr:
            slr_output = self.slr_layer(encoder_output)  # [b,l,2h]
            link_start_hidden, link_end_hidden = torch.chunk(slr_output, 2, dim=-1)
            link_scores = calc_link_score(link_start_hidden, link_end_hidden)  # b,l-1
            pooling_type = 'softmin'
            logsumexp_temp = 0.3
            self.refined_scores = calc_refined_mat_tensor(link_scores, pooling_type=pooling_type, temp=logsumexp_temp)  # b,l,l,1

        return output_per_task_lst

    def task_layer_forward3(self, encoder_output, use_task_embed=True, use_gumbel_softmax=True, gumbel_tasks=None, ep=None):
        output_per_task_lst = []  # list of [b,l,h~]
        for task_id in range(self.num_tasks):
            if use_task_embed:
                gate_logit = self.task_embed[task_id]  # [1024]
                # calc gate [emb_size]
                if use_gumbel_softmax:  # gate binary vector {0,1} with random sample. logits is distribution
                    if task_id in gumbel_tasks:  # when train specific task, which enable grad back propagation
                        # if ep == 0:
                        #     gate = torch.ones_like(gate_logit)
                        #     self.gate_tensor_lst[task_id] = gate_logit
                        # elif ep == 1:
                        #     gate = gumbel_sigmoid(gate_logit, hard=False, use_gumbel=False, tau=1/3, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # elif ep > 1:
                        #     gate = gumbel_sigmoid(gate_logit, hard=True, use_gumbel=True, tau=1/3, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # if ep <= 6:
                        #     gate = gumbel_sigmoid(gate_logit, hard=True, generator=self.gumbel_generator)  # random output
                        #     self.gate_tensor_lst[task_id] = gate
                        # else:
                        #     gate = (gate_logit > 0.).float()
                        #     self.gate_tensor_lst[task_id] = gate
                        if ep is not None:
                            gate = (gate_logit >= 0.).float()
                            self.gate_tensor_lst[task_id] = gate

                    else:
                        # 这种方式不会传梯度到gate_logit也就是task_embed去。
                        gate = (gate_logit > 0.).float()  # deterministic output. when is not training, which unable grad back propagation
                    # ipdb.set_trace()
                else:
                    gate = torch.sigmoid(gate_logit)  # [0~1] prob

                input_per_task = encoder_output * gate[None, None, :]
            else:
                input_per_task = encoder_output

            output_per_task = self.task_layers[task_id](input_per_task)
            output_per_task_lst.append(output_per_task)
        output_per_task_lst = torch.cat(output_per_task_lst, dim=-1)  # [b,l,h+]
        return output_per_task_lst

    def compute_offsets(self, task_id, mode='train'):
        ofs_s, ofs_e = self.taskid2offset[task_id]
        if mode == 'train':
            pass
        if mode == 'test':
            ofs_s = 0  # 从累计最开始的task算起
        return int(ofs_s), int(ofs_e)

    def dist(self, x, y, dim, normalize=False):
        if normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        if self.method == 'dot':
            sim = (x * y).sum(dim)
        elif self.method == 'euclidean':
            sim = -(torch.pow(x - y, 2)).sum(dim)
        elif self.method == 'cosine':
            sim = F.cosine_similarity(x, y, dim=dim)
        return sim / self.tau

    def get_contrastive_logits(self, hidden_states, inputs, valid_mask, target_classes):
        class_indexs = [self.tokenizer.get_vocab()[tclass] for tclass in target_classes]

        class_rep = []
        for iclass in class_indexs:
            class_rep.append(torch.mean(hidden_states[inputs.eq(iclass), :].view(-1, hidden_states.size(-1)), 0))

        class_rep = torch.stack(class_rep).unsqueeze(0)
        token_rep = hidden_states[valid_mask != self.tokenizer.pad_token_id, :].view(-1,
                                                                                     hidden_states.size(-1)).unsqueeze(
            1)

        logits = self.dist(class_rep, token_rep, -1)

        return logits.view(-1, len(target_classes))

    def calc_mse_loss(self, pred_ids, true_ids, seq_len, label_tags, mse_mask, kl_mask):
        # 确保输入是浮点型
        pred_ids = pred_ids.float()
        true_ids = true_ids.float()
        seq_len_mask = sequence_mask(seq_len)  # b,l
        # mse_mask = (label_tags == self.loader.datareader.tag2id.any2id['O'])
        # 计算均方误差损失
        loss_fn = nn.MSELoss(reduction='none')
        # scale_factor = 10000  # 可以调整
        # pred_ids /= scale_factor
        # true_ids /= scale_factor
        mse_loss = loss_fn(pred_ids, true_ids)
        mse_loss = mse_loss * seq_len_mask.unsqueeze(-1)  # 除去PAD
        mse_loss = mse_loss * kl_mask.unsqueeze(-1)    # 除去当前任务实体
        mse_loss = mse_loss * mse_mask.unsqueeze(-1)   # 出去之前任务实体

        # num_mse = torch.sum(torch.logical_and(seq_len_mask, mse_mask))
        num_mse = torch.sum(torch.logical_and(torch.logical_and(seq_len_mask, kl_mask), mse_mask))

        if num_mse == 0.:
            mse_loss = 0.
        else:
            mse_loss = mse_loss.sum() / num_mse

        return mse_loss

    def calc_kl_loss(self, predict, target, seq_len_mask, kl_mask):
        target = torch.nan_to_num(target, nan=-1e8)   # 不加这个会使得kl_loss为nan
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(predict, dim=-1),
            torch.nn.functional.log_softmax(target, dim=-1),
            log_target=True,
            reduction='none')  # [b,l,dim]

        kl_loss = kl_loss.sum(-1)  # [b,l]
        kl_loss = kl_loss * seq_len_mask
        kl_loss = kl_loss * kl_mask

        num_kl = torch.sum(torch.logical_and(seq_len_mask, kl_mask))
        if num_kl == 0.:
            kl_loss = 0.
        else:
            kl_loss = kl_loss.sum() / num_kl
        # print(num_kl, kl_loss)
        # print(seq_len_mask.sum())
        return kl_loss  # [b,l]

    def calc_ce_loss(self, target, predict, seq_len_mask, ce_mask, crr_offe):
        # target [b,l]  predict [b,l,tag]
        # # 创建权重张量，为O类token赋予更高的权重
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # weights = torch.ones(crr_offe).to(device)
        # # O类token ID设为较高权重
        # for token in self.tokenizer.get_vocab():
        #     if token not in ['I-PER', 'I-MISC', 'I-ORG', 'I-LOC']:
        #         weights[self.tokenizer.convert_tokens_to_ids(token)] = 2.0
        # ce_loss_layer = nn.CrossEntropyLoss(weight=weights, reduction='none')
        ce_loss_layer = nn.CrossEntropyLoss(reduction='none')
        ce_loss = ce_loss_layer(predict.transpose(1, 2), target)  # [b,ent,l] [b,l] -> [b,l]
        ce_loss = ce_loss * seq_len_mask
        ce_loss = ce_loss * ce_mask
        # return span_loss.mean()  # [b,l]

        num_ce = torch.sum(torch.logical_and(seq_len_mask, ce_mask)) # 实际预测的token个数
        if num_ce == 0.:
            ce_loss = 0.
        else:
            ce_loss = ce_loss.sum() / num_ce
        # print(num_ce, ce_loss)
        return ce_loss  # [b,l]

    def calc_loss(self, labels, distilled_logits, crr_prediction_scores, seq_len, curr_task_id, num_ents_per_task,
                  crr_offe):
        # task_ent_output  # :截至当前任务的 0:offset_e
        seq_len_mask = sequence_mask(seq_len)  # b,l
        # ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        if curr_task_id == 0:  # 第一个任务
            ce_mask = seq_len_mask  # ce_mask [b,l] 哪些是要计算ce_loss的 第一个任务所有都要。
            ce_loss = self.calc_ce_loss(labels, crr_prediction_scores, seq_len_mask, ce_mask, crr_offe)
            kl_loss = torch.tensor(0.)
            mse_loss = torch.tensor(0.)

            # # 提取原始句子中的实体词表征
            # original_entity_reps = encoder_output[:, original_entity_positions,
            #                        :]  # original_entity_positions表示原始句子中实体词的位置
            # # 提取模板中的锚点词表征
            # anchor_entity_reps = encoder_output[:, template_entity_positions, :]  # template_entity_positions表示模板中锚点词的位置
            # # 提取非实体词的表征
            # non_entity_reps = encoder_output[:, non_entity_positions, :]  # non_entity_positions表示非实体词的位置
            # # 计算正样本（实体词与对应锚点词）的相似度
            # pos_similarity = F.cosine_similarity(original_entity_reps, anchor_entity_reps, dim=-1)
            # # 计算负样本（实体词与非实体词）的相似度
            # neg_similarity = F.cosine_similarity(original_entity_reps.unsqueeze(1), non_entity_reps.unsqueeze(0),
            #                                      dim=-1)
            # # 使用 InfoNCE 损失
            # contrastive_loss = -torch.log(
            #     F.softmax(pos_similarity, dim=-1) / (
            #                 F.softmax(pos_similarity, dim=-1) + F.softmax(neg_similarity, dim=-1)))
            # contrastive_loss = contrastive_loss.mean()

        else:  # 后续任务
            # ce_mask = torch.logical_and(ce_target >= ofs, ce_target < ofe).float()  # 之后的任务当前token属于新任务的ent时要
            ce_mask = torch.logical_and(labels >= crr_offe - num_ents_per_task[curr_task_id],
                                        labels < crr_offe).float()  # 之后的任务当前token属于新任务的ent时要
            # predict = ent_output[:, :, :ofe]  # 取13个输出维度的前5维度（即O、B-ORG、I-ORG、B-PER、I-PER）
            ce_loss = self.calc_ce_loss(labels, crr_prediction_scores, seq_len_mask, ce_mask, crr_offe)

            kl_mask = 1. - ce_mask
            # kl_predict = ent_output[:, :, :ofs]
            kl_predict = crr_prediction_scores  # 用 小值来补新标签对应的B和I
            # print(kl_target.shape)
            # kl_target = torch.nn.functional.pad(distilled_logits, (0, ofe - ofs), mode='constant', value=-1e8)

            kl_target = [torch.nn.functional.pad(tensor, (0, num_ents_per_task[curr_task_id]),
                                                 mode='constant', value=torch.tensor(-1e8, dtype=tensor.dtype)) for
                         tensor in distilled_logits]
            # -1e8填充为等长
            max_len = max(len(seq) for seq in kl_target)
            kl_target = nn.utils.rnn.pad_sequence(
                [torch.cat([seq, torch.full((max_len - len(seq), kl_target[0].size(1)), fill_value=-1e8,
                                            device=kl_target[0].device)]) for seq in kl_target],
                batch_first=True)
            # labels = torch.stack([nn.functional.pad(seq, (0, max_len - len(seq)), value=-1e8) for seq in kl_target])
            # kl_target = torch.stack(kl_target)

            # print(kl_target.shape)
            # print(kl_predict.shape)
            # ipdb.set_trace()
            kl_loss = self.calc_kl_loss(kl_predict, kl_target, seq_len_mask, kl_mask)

        return ce_loss, kl_loss

    def calc_3_loss(self, labels, distilled_logits, crr_prediction_scores, seq_len, curr_task_id, num_ents_per_task,
                  crr_offe, bert_input_ids, batch_gold_label_ids):
        # task_ent_output  # :截至当前任务的 0:offset_e
        seq_len_mask = sequence_mask(seq_len)  # b,l
        # ofs, ofe = self.task_offset_lst[task_id]  # 当前任务的offset是
        if curr_task_id == 0:  # 第一个任务
            ce_mask = seq_len_mask  # ce_mask [b,l] 哪些是要计算ce_loss的 第一个任务所有都要。
            ce_loss = self.calc_ce_loss(labels, crr_prediction_scores, seq_len_mask, ce_mask, crr_offe)
            kl_loss = torch.tensor(0.)
            mse_loss = torch.tensor(0.)

        else:  # 后续任务
            # ce_mask = torch.logical_and(ce_target >= ofs, ce_target < ofe).float()  # 之后的任务当前token属于新任务的ent时要
            ce_mask = torch.logical_and(labels >= crr_offe - num_ents_per_task[curr_task_id],
                                        labels < crr_offe).float()  # 之后的任务当前token属于新任务的ent时要
            # predict = ent_output[:, :, :ofe]  # 取13个输出维度的前5维度（即O、B-ORG、I-ORG、B-PER、I-PER）
            ce_loss = self.calc_ce_loss(labels, crr_prediction_scores, seq_len_mask, ce_mask, crr_offe)

            kl_mask = 1. - ce_mask
            # kl_predict = ent_output[:, :, :ofs]
            kl_predict = crr_prediction_scores  # 用 小值来补新标签对应的B和I
            # print(kl_target.shape)
            # kl_target = torch.nn.functional.pad(distilled_logits, (0, ofe - ofs), mode='constant', value=-1e8)

            kl_target = [torch.nn.functional.pad(tensor, (0, num_ents_per_task[curr_task_id]),
                                                 mode='constant', value=torch.tensor(-1e8, dtype=tensor.dtype)) for
                         tensor in distilled_logits]
            # -1e8填充为等长
            max_len = max(len(seq) for seq in kl_target)
            kl_target = nn.utils.rnn.pad_sequence(
                [torch.cat([seq, torch.full((max_len - len(seq), kl_target[0].size(1)), fill_value=-1e8,
                                            device=kl_target[0].device)]) for seq in kl_target],
                batch_first=True)
            # labels = torch.stack([nn.functional.pad(seq, (0, max_len - len(seq)), value=-1e8) for seq in kl_target])
            # kl_target = torch.stack(kl_target)

            # print(kl_target.shape)
            # print(kl_predict.shape)
            # ipdb.set_trace()
            kl_loss = self.calc_kl_loss(kl_predict, kl_target, seq_len_mask, kl_mask)

            kl_ids = kl_target.argmax(dim=-1)
            mse_mask = torch.logical_or(kl_ids < 28996,  # TODO
                                                  kl_ids >= crr_offe - num_ents_per_task[curr_task_id]
            )

            # bert_pred_ids = crr_prediction_scores.argmax(dim=-1)  # 这个值不能进行梯度求导
            bert_pred_ids = F.softmax(crr_prediction_scores, dim=-1)
            # 将 bert_input_ids 转换为 one-hot 编码
            true_ids_one_hot = torch.nn.functional.one_hot(bert_input_ids,
                                                           num_classes=crr_offe).float()  # (32, 56, 28997)

            mse_loss = self.calc_mse_loss(bert_pred_ids, true_ids_one_hot, seq_len, batch_gold_label_ids, mse_mask, kl_mask)


        return ce_loss, kl_loss, mse_loss

    def calc_f1(self, batch_span_pred, batch_span_tgt):
        # pred batch_span_pred [*,ent], label batch_span_tgt [*,ent]
        batch_span_pred = (batch_span_pred > 0).int()  # [*,ent] logits before sigmoid
        # calc f1
        num_pred = torch.sum(batch_span_pred)
        num_gold = torch.sum(batch_span_tgt)
        tp = torch.sum(batch_span_tgt * batch_span_pred)
        f1 = torch.tensor(1.) if num_gold == num_pred == 0 else 2 * tp / (num_gold + num_pred + 1e-12)
        return f1.item(), (num_gold.item(), num_pred.item(), tp.item())

    def calc_mlm_kl_loss(self, predicted_logits, target_logits):
        predicted_probs = torch.nn.functional.softmax(predicted_logits, dim=-1)
        target_probs = torch.nn.functional.softmax(target_logits, dim=-1)

        kl_loss = torch.nn.functional.kl_div(predicted_probs.log(), target_probs, reduction='batchmean')

        return kl_loss

    def kl_divergence(self, p, q):
        # return torch.nn.functional.kl_div(torch.nn.functional.log_softmax(p, dim=1), torch.softmax(q, dim=1),
        #                                   reduction='batchmean')
        eps = 1e-10  # 很小的常数，防止除以零
        p = torch.clamp_min(p, eps)
        q = torch.clamp_min(q, eps)
        return torch.nn.functional.kl_div(torch.nn.functional.log_softmax(p, dim=1), torch.softmax(q, dim=1),
                                          reduction='batchmean')

    def calc_kl_loss_mse(self, batch_target_lst_distilled, batch_predict_lst_need_distill, task_id):
        ofs_s, ofs_e = self.compute_offsets(task_id)
        bsz = len(batch_target_lst_distilled)
        # kl_losses = []
        batch_kl_loss = 0.
        for bdx in range(bsz):  # 只蒸馏当前的样本 忽略记忆库的
            kl_tgt = batch_target_lst_distilled[bdx][:, :ofs_s]  # prob

            pred_need_distill = batch_predict_lst_need_distill[bdx][:, :ofs_s]
            mse_loss = self.mse_loss_layer(pred_need_distill.sigmoid(), kl_tgt)  # [num_spans, ent]

            mse_loss = torch.mean(mse_loss, -1)  # over ent
            mse_loss = torch.sum(mse_loss, -1)  # over spans
            # mse_loss = torch.mean(mse_loss, -1)  # over spans

            # kl_losses.append(kl_loss)
            # self.kl_loss = sum(kl_losses) / len(kl_losses)
            batch_kl_loss += mse_loss

        return batch_kl_loss / bsz

    def take_loss(self, task_id, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e = self.compute_offsets(task_id)
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头  只计算当前任务的loss
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def take_multitask_loss(self, batch_task_id, batch_predict_lst, batch_target_lst, f1_meaner=None):
        """multiple task of example"""  # batch_predict_lst,batch_target_lst: list of [num_spans, ent]
        losses = []
        for bdx, task_id in enumerate(batch_task_id):  # 以样本为单位计算loss 该batch有多个任务
            ofs_s, ofs_e = self.compute_offsets(task_id)
            span_loss = self.calc_loss(batch_predict_lst[bdx][:, ofs_s:ofs_e], batch_target_lst[bdx][:, ofs_s:ofs_e])  # 只计算对应task的头
            f1, f1_detail = self.calc_f1(batch_predict_lst[bdx][:, ofs_s:ofs_e], batch_target_lst[bdx][:, ofs_s:ofs_e])
            losses.append(span_loss)
            if f1_meaner is not None:
                f1_meaner.add(*f1_detail)
        loss = sum(losses) / len(losses)
        return loss

    def eval_forward(self, inputs_dct, task_id, mode='train'):
        # 用于eval
        seq_len = inputs_dct['ori_seq_len']
        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=[])
        batch_span_pred = self.span_matrix_forward(task_layer_output, seq_len)  # [bsz*num_spans, ent]
        batch_predict_lst = torch.split(batch_span_pred, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        ofs_s, ofs_e = self.compute_offsets(task_id, mode=mode)  # make sure we predict classes within the current task

        f1, detail_f1, span_loss, kl_loss = None, None, None, None
        if 'batch_span_tgt' in inputs_dct:  # if label had passed into
            batch_span_tgt = inputs_dct['batch_span_tgt']  # [bsz*num_spans, ent]
            f1, detail_f1 = self.calc_f1(batch_span_pred[:, ofs_s:ofs_e], batch_span_tgt[:, ofs_s:ofs_e])

            span_loss = self.take_loss(task_id, batch_span_pred, batch_span_tgt, bsz=len(seq_len))  # 默认是curr(train)
        if self.args.use_distill and task_id > 0 and inputs_dct.get('batch_span_tgt_lst_distilled', None):
            kl_loss = self.calc_kl_loss(inputs_dct['batch_span_tgt_lst_distilled'], batch_predict_lst, task_id)  # 默认是so_far-curr

        return batch_span_pred[:, ofs_s:ofs_e], f1, detail_f1, span_loss, kl_loss

    def get_labels(self, predictions, references, tokens, task_id=None, emissions=None, viterbi_decoder=None,
                   mode=None):

        use_crf = True if emissions is not None else False
        # Transform predictions and references tensos to numpy arrays
        if self.args.device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
            x_tokens = tokens.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()
            x_tokens = tokens.detach().cpu().clone().tolist()
            if use_crf:
                emissions = emissions.detach().cpu().clone().numpy()

        label_token_map = {item: item for item in self.loader.ori_label_token_map}
        label_token_to_id = {label: self.tokenizer.convert_tokens_to_ids(label_token) for label, label_token in
                             label_token_map.items()}
        label_token_id_to_label = {idx: label for label, idx in label_token_to_id.items()}
        if use_crf:
            # The viterbi decoding algorithm

            out_label_ids = y_true
            preds = y_pred

            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            emissions_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]

            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != -100:
                        out_label_list[i].append(self.loader.datareader.id2tag.any2id[out_label_ids[i][j]])
                        emissions_list[i].append(emissions[i][j])
                        preds_list[i].append(label_token_id_to_label[preds[i][j]] if preds[i][
                                                                                         j] in label_token_id_to_label.keys() else 'O')

            preds_list = [[] for _ in range(out_label_ids.shape[0])]
            for i in range(out_label_ids.shape[0]):
                sent_scores = torch.tensor(emissions_list[i])
                sent_len, n_label = sent_scores.shape
                sent_probs = torch.nn.functional.softmax(sent_scores, dim=1)
                start_probs = torch.zeros(sent_len) + 1e-6
                sent_probs = torch.cat((start_probs.view(sent_len, 1), sent_probs), 1)
                feats = viterbi_decoder.forward(torch.log(sent_probs).view(1, sent_len, n_label + 1))
                vit_labels = viterbi_decoder.viterbi(feats)
                vit_labels = vit_labels.view(sent_len)
                vit_labels = vit_labels.detach().cpu().numpy()
                for label in vit_labels:
                    preds_list[i].append(self.loader.datareader.id2tag.any2id[label - 1])

            true_predictions = preds_list
            true_labels = out_label_list
            ori_tokens = [
                [self.tokenizer.convert_ids_to_tokens(t) for (p, l, t) in zip(pred, gold_label, token) if l != -100]
                for pred, gold_label, token in zip(y_pred, y_true, x_tokens)
            ]

        else:
            # Remove ignored index (special tokens)
            # Here we only use the first token of each word for evaluation.
            true_predictions = [
                [label_token_id_to_label[p] if p in label_token_id_to_label.keys() else 'O' for (p, l) in
                 zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            # id_to_label = {id: label for label, id in self.label_to_id.items()}
            id_to_label = {id: label for label, id in self.loader.datareader.tag2id.any2id.items()}
            true_labels = [
                [id_to_label[l] for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]

            ori_tokens = [
                [self.tokenizer.convert_ids_to_tokens(t) for (p, l, t) in zip(pred, gold_label, token) if l != -100]
                for pred, gold_label, token in zip(y_pred, y_true, x_tokens)
            ]

        def switch_to_BIO(labels):
            past_label = 'O'
            labels_BIO = []
            for label in labels:
                if label.startswith('I-') and (past_label == 'O' or past_label[2:] != label[2:]):
                    labels_BIO.append('B-' + label[2:])
                else:
                    labels_BIO.append(label)
                past_label = label
            return labels_BIO

        # Turn the predictions into required label schema.
        # if args.label_schema == "IO" and args.eval_label_schema == 'BIO':
        true_predictions = list(map(switch_to_BIO, true_predictions))
        # if args.eval_label_schema == 'IO':
        # true_labels = [['I-{}'.format(l[2:]) if l !='O' else 'O' for l in label] for label in true_labels]
        # 只保留当前任务相关标签
        if mode == 'test':
            crr_task_labels_lst = self.loader.entity_task_lst[0:task_id + 1]
            crr_task_labels_lst = [task for task_list in crr_task_labels_lst for task in task_list]
        else:
            crr_task_labels_lst = self.loader.entity_task_lst[task_id]  # ['ORG']
        matching_labels = ['I-' + task_label for task_label in crr_task_labels_lst] + ['B-' + task_label for task_label
                                                                                       in crr_task_labels_lst]
        crr_labels = [['O' if label not in matching_labels else label for label in sublist] for sublist in
                      true_labels]  # TODO 不要写死
        crr_true_predictions = [['O' if label not in matching_labels else label for label in sublist] for sublist in
                                true_predictions]  # TODO 不要写死

        return crr_true_predictions, crr_labels, ori_tokens

    def calculate_bio_f1(self, batch_bio_pred, batch_bio_tgt):
        num_pred = sum(tag.startswith('B-') for tags in batch_bio_pred for tag in tags)
        num_gold = sum(tag.startswith('B-') for tags in batch_bio_tgt for tag in tags)

        tp = sum(1 for pred_tags, gold_tags in zip(batch_bio_pred, batch_bio_tgt)
                 for pt, gt in zip(pred_tags, gold_tags) if gt != 'O' and pt == gt)
        fp = sum(1 for pred_tags, gold_tags in zip(batch_bio_pred, batch_bio_tgt)
                 for pt, gt in zip(pred_tags, gold_tags) if pt != 'O' and gt == 'O')
        fn = sum(1 for pred_tags, gold_tags in zip(batch_bio_pred, batch_bio_tgt)
                 for pt, gt in zip(pred_tags, gold_tags) if gt != 'O' and pt == 'O')

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        return f1, (num_gold, num_pred, tp)

    def mlm_eval_forward(self, inputs_dct, task_id, mode='train'):
        # 用于eval
        batch_entlm_input = {}
        batch_entlm_input['input_ids'] = inputs_dct['input_ids']
        batch_entlm_input['token_type_ids'] = inputs_dct['bert_token_type_ids']
        batch_entlm_input['attention_mask'] = inputs_dct['bert_attention_mask']

        all_task_labels = inputs_dct['batch_tag_token_onehot']  # 这里应该是全部任务标签
        ofs_s, ofs_e = self.compute_offsets(task_id, mode=mode)
        selected_columns = [task_labels[:, ofs_s:ofs_e] for task_labels in all_task_labels]
        crren_task_labels = []
        for i in selected_columns:
            # 找到每个位置的最大值
            max_values, _ = torch.max(i, dim=1, keepdim=True)
            # 对比每个位置，若不同，则将其值统一为该位置的最大值
            # updated_tensor = torch.where(i != max_values, max_values, i)
            crren_task_labels.append(max_values)

        combined_labels = torch.cat([column.unsqueeze(0) for column in crren_task_labels], dim=0)
        crren_task_labels = combined_labels.view(combined_labels.size(0), -1)  # 当前任务的黄金标签batch
        inputs_dct['crren_task_labels'] = crren_task_labels.to(torch.long)  # torch.int64的类型才能进行loss

        batch_entlm_input['labels'] = inputs_dct['crren_task_labels']  # TODO 这一步多余了，可以不要
        batch_entlm_input['ori_labels'] = inputs_dct['batch_gold_label_ids']  # 当前任务的黄金标签的索引batch
        ner_label = batch_entlm_input.pop('ori_labels', 'not found ner_labels')
        batch_entlm_outputs = self.bert_layer(**batch_entlm_input, output_hidden_states=True)
        # TODO 这里需不需要加一个(29002,29002)的FFNN?不需要，因为(cls): BertOnlyMLMHead层已经根据真实29002维label由768转为29002了
        batch_entlm_logits = batch_entlm_outputs.logits  # 这个值是bert将每个token（如‘world’这个token）编码为29002维的向量，如：[ -7.9981,  -7.9188,  -8.0607,  ...,  -1.7187,   0.4554,   0.2795]，其中-7.9981表示‘world’被编码后，预测为[PAD]（因为[PAD]是词表第0个索引）的打分值
        crr_offe = len(self.tokenizer) - sum(self.num_ents_per_task) + sum(self.num_ents_per_task[:task_id + 1])
        batch_entlm_logits = batch_entlm_logits[:, :, :crr_offe]  # 这个值是bert将每个token（如‘world’这个token）编码为29002维的向量，如：[ -7.9981,  -7.9188,  -8.0607,  ...,  -1.7187,   0.4554,   0.2795]，其中-7.9981表示‘world’被编码后，预测为[PAD]（因为[PAD]是词表第0个索引）的打分值

        label_token_map = {item: item for item in self.loader.ori_label_token_map}
        label_token_to_id = {label: self.tokenizer.convert_tokens_to_ids(label_token) for label, label_token in
                             label_token_map.items()}
        # 获取当前任务对应的实体类别
        crr_entity = [self.loader.datareader.id2ent.any2id[task_id]]
        # 通过实体类别找到对应的索引值
        index_values = [label_token_to_id[f"I-{ent_type}"] for ent_type in crr_entity]
        # 将索引值转换为 tensor
        label_id_list = torch.tensor(index_values, device=self.args.device)
        # label_token_id_to_label = {idx: label for label, idx in label_token_to_id.items()}
        # id_to_label = self.loader.datareader.id2tag.any2id
        # label_id_list = torch.tensor([label_token_to_id[id_to_label[i]] for i in range(len(id_to_label)) if
        #                               i != 0 and not id_to_label[i].startswith("B-")], dtype=torch.long, device=self.args.device)
        if self.args.use_crf:
            # abstract_transitions = get_abstract_transitions(args.crf_raw_path, "train")
            abstract_transitions = [0.6706787265864255, 0.32932127341357453, 0.9210515229388718, 0.0789484770611282,
                                    0.7267836556936083, 0.2729601639554246, 0.0002561803509670808]
            viterbi_decoder = ViterbiDecoder(len(['I-MISC', 'O']) + 1,
                                             abstract_transitions, 0.05)

            probs = torch.softmax(batch_entlm_logits, -1)
            emissions = probs[:, :, label_id_list]
            O_emissions = probs[:, :, :label_id_list.min().data].max(-1)[0].unsqueeze(-1)
            emissions = torch.cat([O_emissions, emissions], dim=-1)

        batch_entlm_prob, _ = torch.max(F.softmax(batch_entlm_logits, dim=-1), dim=-1, keepdim=True)
        '''
        batch_entlm_logits这个值是bert将每个token（如‘world’这个token）编码为29002维的向量，
        如：[ -7.9981,  -7.9188,  -8.0607,  ...,  -1.7187,   0.4554,   0.2795]，
        其中-7.9981表示‘world’被编码后，预测为[PAD]（因为[PAD]是词表第0个索引）的打分值；
        F.softmax(batch_entlm_logits.squeeze(0)表示将上述这个预测得分，通过softmax函数将每个token的得分转换为概率，
        如：[0.0000,   0.0000,    0.0000,  ...,    0.0000,   0.0002,   0.0002]，
        其中第一个0.0000表示‘world’被编码后，预测为[PAD]的概率；（因为world属于O类，因此我们想让
        它预测为本身词，即world，的概率更高些，如果world属于PER类，那我们希望它预测为I-PER（已添加
        进词表）的概率更高些）
        最后一步torch.max()是沿着最后一个维度取得最大值，如[0.0559]，即词表world位置的概率值

        总之，简要概括batch_entlm_prob这个值的含义为：非实体词预测为它本身的概率，实体词预测为类枢纽词的概率
        '''
        predictions = batch_entlm_logits.argmax(dim=-1)  # 对每个输入位置，选择具有最高概率的token词表索引

        labels = ner_label
        token_labels = batch_entlm_input.pop("input_ids")

        # 去除CLS、SEP、以及##ed这类子词
        filtered_prob = []
        for i in range(labels.size(0)):
            # 获取当前句子的 logits 和 labels
            prob_i = batch_entlm_prob[i]  # (34,1)
            labels_i = labels[i]  # (34)

            # 使用非 -100 的位置进行过滤
            # filtered_prob_i = prob_i[:len(labels_i)][labels_i != -100, :]
            filtered_prob_i = prob_i

            # 将过滤后的添加到结果列表
            filtered_prob.append(filtered_prob_i)
        batch_entlm_prob_list = filtered_prob

        # 去掉CLS、SEP、以及##ed这类子词
        # filtered_logits = []
        # for i in range(labels.size(0)):
        #     # 获取当前句子的 logits 和 labels
        #     logits_i = mlm_outputs.logits[i]  # (34,29002)
        #     labels_i = labels[i]  # (34)
        #
        #     # 使用非 -100 的位置进行过滤
        #     filtered_logits_i = logits_i[labels_i != -100, :]
        #
        #     # 将过滤后的 logits 添加到结果列表
        #     filtered_logits.append(filtered_logits_i)

        filtered_logits = []
        for i in range(labels.size(0)):
            # 获取当前句子的 logits 和 labels
            logits_i = batch_entlm_logits[i]  # (34,1)
            labels_i = labels[i]  # (34)

            # 使用非 -100 的位置进行过滤
            # filtered_logits_i = logits_i[:len(labels_i)][labels_i != -100, :]
            filtered_logits_i = logits_i

            # 将过滤后的添加到结果列表
            filtered_logits.append(filtered_logits_i)
        batch_entlm_logits_list = filtered_logits
        if self.args.use_crf:
            preds, refs, tokens = self.get_labels(predictions, labels, token_labels, task_id, emissions,
                                                  viterbi_decoder, mode=mode)
        else:
            preds, refs, tokens = self.get_labels(predictions, labels, token_labels, task_id, mode=mode)
        f1, f1_detail = self.calculate_bio_f1(preds, refs)
        # kl_loss = None
        # if self.args.use_distill and task_id > 0 and inputs_dct.get('batch_span_tgt_lst_distilled', None):
        #     kl_loss = self.calc_kl_loss(inputs_dct['batch_span_tgt_lst_distilled'], batch_predict_lst, task_id)  # 默认是so_far-curr

        return preds, refs, tokens, batch_entlm_prob_list, batch_entlm_logits_list, f1, f1_detail

    def forward(self, *args, **kwargs):   # 评估时调用此方法
        return self.mlm_eval_forward(*args, **kwargs)   # rz+

    def observe_mlm(self, inputs_dct, task_id, f1_meaner, ep=None):   # 训练时调用此方法
        bert_input_ids = inputs_dct['input_ids']
        bert_token_type_ids = inputs_dct['bert_token_type_ids']
        bert_attention_mask = inputs_dct['bert_attention_mask']
        seq_len = inputs_dct['seq_len']  # 子词个数
        batch_word_mask = inputs_dct['batch_word_mask_lst']
        batch_wo_pad_len = inputs_dct['batch_wo_pad_len']
        # batch_ner_exm = inputs_dct['batch_ner_exm']
        ori_seq_len = inputs_dct['ori_seq_len']  # 原始词个数
        batch_ori_2_tok = inputs_dct['batch_ori_2_tok']
        # batch_tag_ids = inputs_dct['batch_tag_ids']
        # batch_tag_token = inputs_dct['batch_tag_token']
        all_task_labels = inputs_dct['batch_tag_token_onehot']  # 所有6个任务的黄金标签
        batch_gold_label_ids = inputs_dct['batch_gold_label_ids']

        ofs_s, ofs_e = self.compute_offsets(task_id)
        selected_columns = [task_labels[:, ofs_s:ofs_e] for task_labels in all_task_labels]  # 当前任务的黄金标签
        combined_labels = torch.cat([column.unsqueeze(0) for column in selected_columns], dim=0)  # 当前任务的黄金标签batch
        crren_task_labels = combined_labels.view(combined_labels.size(0), -1)  # 当前任务的黄金标签batch
        labels = crren_task_labels.to(torch.long)  # torch.int64的类型才能进行loss
        # batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        batch_distilled_task_ent_output = inputs_dct.get('batch_distilled_task_ent_output', None)  # list of sigmoided prob [num_spans, ent]

        # self.total_loss = 0.
        # self.mlm_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.
        self.mse_loss = 0.

        encoder_output = self.entlm_forward(bert_input_ids, bert_token_type_ids, bert_attention_mask,
                                            seq_len, batch_ori_2_tok, batch_word_mask, batch_wo_pad_len)   # (32,40,768)，其中40表示的是句子中词个数（不是子词个数）

        crr_offe = len(self.tokenizer) - sum(self.num_ents_per_task) + sum(self.num_ents_per_task[:task_id + 1])
        crr_prediction_scores = encoder_output[:, :, :crr_offe]

        # 去除labels的[CLS]和[SEP] TODO 从datareader里改吧
        # labels_lst = [t for t in labels]  # split along batch
        # for i, t in enumerate(labels_lst):  # iter along batch
        #     # tensor [len, hid]
        #     labels_lst[i] = torch.cat([t[1: 1 + seq_len[i]], t[2 + seq_len[i]:]], 0)
        # labels = torch.stack(labels_lst, 0)  # stack along batch
        #
        # if batch_ori_2_tok.shape[0]:  # 只取子词的第一个字  ENG
        #     labels_lst = [t for t in labels]
        #     for bdx, t in enumerate(labels_lst):
        #         ori_2_tok = batch_ori_2_tok[bdx]
        #         ori_2_tok = ori_2_tok[:ori_seq_len[bdx]]
        #         labels_lst[bdx] = labels_lst[bdx][ori_2_tok]
        #     # -100填充
        #     max_len = max(len(seq) for seq in labels_lst)
        #     labels = torch.stack(
        #         [nn.functional.pad(seq, (0, max_len - len(seq)), value=-100) for seq in labels_lst])

        # ce_loss, kl_loss = self.calc_loss(labels, batch_distilled_task_ent_output, crr_prediction_scores, ori_seq_len,
        ce_loss, kl_loss = self.calc_loss(labels, batch_distilled_task_ent_output, crr_prediction_scores, batch_wo_pad_len,
                                          task_id, self.num_ents_per_task, crr_offe)
        # ce_loss, kl_loss, mse_loss = self.calc_3_loss(labels, batch_distilled_task_ent_output, crr_prediction_scores, batch_wo_pad_len,
        #                                   task_id, self.num_ents_per_task, crr_offe, bert_input_ids, batch_gold_label_ids)  # 查看label这个tensor是否包含28996（PER）：torch.any(labels == 28996).item()

        self.ce_loss = ce_loss
        self.kl_loss = kl_loss
        # self.mse_loss = mse_loss
        self.total_loss = self.ce_loss + self.kl_loss
        # self.total_loss = self.ce_loss + self.kl_loss + self.mse_loss
        # self.total_loss = self.ce_loss ** 2 + self.kl_loss
        # self.total_loss = self.ce_loss**2 + 5 * self.kl_loss**2

        self.opt.zero_grad()   # 将模型参数的梯度归零,以避免梯度的累积
        self.total_loss.backward()   # 对总的损失进行反向传播,计算模型参数关于损失函数的梯度。

        if self.grad_clip is None:   # self.grad_clip这个参数在onto上为1.0，在fewnerd上为5.0
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)  # 梯度裁剪,目的是限制梯度的范数，以防止梯度爆炸的问题（加这个很重要）
            # self.total_norm = 0

        self.opt.step()    # 使用优化器来更新模型参数,优化器根据梯度和预定义的优化算法来更新模型的权重,此时模型的参数会根据梯度和学习率进行更新。
        # if ep is not None:
        #     if ep <= 6:
        #         self.lrs.step()
        #     else:
        #         pass
        if self.use_schedual:
            self.lrs.step()     # 学习率调度器根据预定义的规则调整学习率，可以帮助模型更好地收敛到最优解。
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.ce_loss),
                float(self.mse_loss),
                float(self.kl_loss),
                )

    def take_alltask_loss(self, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        loss = self.calc_loss(batch_predict, batch_target)  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict, batch_target)
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def observe_all(self, inputs_dct, f1_meaner, ep=None):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.opt.zero_grad()

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=list(range(self.num_tasks)),
                                                     )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        self.span_loss = self.take_alltask_loss(batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # 这样loss不能按batch平均
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_task_embed:
            for task_id in range(self.num_tasks):
                if self.args.use_gumbel_softmax:
                    curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                else:
                    curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
                sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
                self.sparse_loss += sparse_loss
            self.sparse_loss /= self.num_tasks
            self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )

    def take_so_far_task_loss(self, task_id, batch_predict, batch_target, f1_meaner=None, bsz=None):
        """single task of example"""  # batch_predict,batch_target: [bsz*num_spans, ent]
        ofs_s, ofs_e = self.compute_offsets(task_id, mode='test')
        loss = self.calc_loss(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])  # 只计算对应task的头
        if f1_meaner is not None:
            f1, f1_detail = self.calc_f1(batch_predict[:, ofs_s:ofs_e], batch_target[:, ofs_s:ofs_e])
            f1_meaner.add(*f1_detail)
        if bsz is not None:
            loss = loss / bsz
        return loss

    def observe_non_cl(self, inputs_dct, task_id, f1_meaner):
        batch_exm = inputs_dct['batch_ner_exm']
        batch_length = inputs_dct['ori_seq_len']
        batch_input_pts = inputs_dct['batch_input_pts']

        batch_target = inputs_dct['batch_span_tgt']
        batch_target_lst = inputs_dct['batch_span_tgt_lst']  # list of onehot tensor [num_spans, ent]
        batch_target_lst_distilled = inputs_dct.get('batch_span_tgt_lst_distilled', None)  # list of sigmoided prob [num_spans, ent]
        bsz = batch_length.shape[0]

        self.opt.zero_grad()

        self.total_loss = 0.
        self.span_loss = 0.
        self.sparse_loss = 0.
        self.kl_loss = 0.
        self.entropy_loss = 0.

        encoder_output = self.encoder_forward(inputs_dct)
        task_layer_output = self.task_layer_forward2(encoder_output,
                                                     use_task_embed=self.args.use_task_embed, use_gumbel_softmax=self.args.use_gumbel_softmax,
                                                     gumbel_tasks=list(range(self.num_tasks)),
                                                     )

        batch_predict = self.span_matrix_forward(task_layer_output, batch_length)  # [bsz*num_spans, ent]

        batch_predict_lst = torch.split(batch_predict, (batch_length * (batch_length + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        self.span_loss = self.take_so_far_task_loss(task_id, batch_predict, batch_target, f1_meaner=f1_meaner, bsz=bsz)  # 这样loss不能按batch平均
        self.total_loss += self.span_loss
        # self.total_loss += self.span_loss / (self.span_loss.detach() + 1e-7)

        if self.args.use_task_embed:
            for task_id in range(self.num_tasks):
                if self.args.use_gumbel_softmax:
                    curr_task_gate = self.gate_tensor_lst[task_id]  # [emb_dim] 0-1 binary
                else:
                    curr_task_gate = torch.sigmoid(self.task_embed[task_id])  # [emb_dim]  0~1 prob  # 这样没有sparse_loss
                sparse_loss = torch.norm(curr_task_gate, p=1, dim=-1) / curr_task_gate.shape[0]  # L1范数 need / 1024dim
                self.sparse_loss += sparse_loss
            self.sparse_loss /= self.num_tasks
            self.total_loss += self.sparse_loss
            # self.total_loss += self.sparse_loss / (self.sparse_loss.detach() + 1e-7)

        self.opt.zero_grad()
        self.total_loss.backward()

        if self.grad_clip is None:
            self.total_norm = 0
        else:
            self.total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        self.opt.step()
        if self.use_schedual:
            self.lrs.step()
        # self.curr_lr = self.lrs.get_last_lr()[0]
        self.curr_lr = self.opt.param_groups[0]['lr']

        return (float(self.total_loss),
                float(self.span_loss),
                float(self.sparse_loss),
                float(self.kl_loss),
                )


def get_BIO_transitions_mask(tag2id, id2tag):
    num_tags = len(tag2id)
    all_tag_lst = list(tag2id.keys())
    trans_mask = np.ones([num_tags, num_tags], dtype=np.int)  # 默认全部要mask(不通=1) 不mask则为0
    O_tag_id = tag2id['O']
    trans_mask[O_tag_id, :] = 0  # O能转为任何
    trans_mask[:, O_tag_id] = 0  # 任何都能转为O
    for prev_tag_id in range(num_tags):
        prev_tag = id2tag[prev_tag_id]
        if prev_tag.startswith('B-'):  # B- 只能跟自己的I 和 任何其他B- # TODO 允不允许B1B1I1？
            prev_tag_name = prev_tag.replace('B-', '')
            trans_mask[prev_tag_id][tag2id[f'I-{prev_tag_name}']] = 0
            for rear_tag in all_tag_lst:
                # if rear_tag.startswith('B-') and rear_tag != prev_tag:  # 不允许B1B1I1
                if rear_tag.startswith('B-'):  # 允许B1B1I1
                    trans_mask[prev_tag_id][tag2id[rear_tag]] = 0
        if prev_tag.startswith('I-'):  # 只能跟自己的I 和任何其他B-
            prev_tag_name = prev_tag.replace('I-', '')
            trans_mask[prev_tag_id][tag2id[f'I-{prev_tag_name}']] = 0
            for rear_tag in all_tag_lst:
                if rear_tag.startswith('B-'):
                    trans_mask[prev_tag_id][tag2id[rear_tag]] = 0
    return trans_mask


def viterbi_decode(emissions: torch.FloatTensor,
                   mask: torch.ByteTensor,
                   start_transitions,
                   end_transitions,
                   transitions) -> List[List[int]]:
    # emissions: (batch_size,seq_length,num_tags)
    # mask: (batch_size,seq_length)
    # start_transitions end_transitions [num_tags]
    # transitions [num_tags, num_tags]

    emissions = emissions.transpose(0, 1)
    mask = mask.transpose(0, 1)

    assert emissions.dim() == 3 and mask.dim() == 2
    assert emissions.shape[:2] == mask.shape
    assert emissions.size(2) == start_transitions.size(0) == end_transitions.size(0) == transitions.size(0)
    assert mask[0].all()

    seq_length, batch_size = mask.shape

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag
    for i in range(1, seq_length):
        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = score.unsqueeze(2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = emissions[i].unsqueeze(1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + start_transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        next_score, indices = next_score.max(dim=1)

        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = torch.where(mask[i].unsqueeze(1), next_score, score)
        history.append(indices)

    # End transition score
    # shape: (batch_size, num_tags)
    score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = mask.long().sum(dim=0) - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        _, best_last_tag = score[idx].max(dim=0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list

# Below functions are for SLR (another reseach), not used in this published paper.
def calc_link_score(link_start_hidden, link_end_hidden, fast_impl=True):
    # link_start_hidden [b,l,h]
    # link_end_hidden [b,l,h]
    # return link_dot_prod_scores [b,l-1]
    hidden_size = link_start_hidden.shape[-1]

    if fast_impl:
        # link score 快速计算方式 直接移位相乘再相加(点积)
        link_dot_prod_scores = link_start_hidden[:, :-1, :] * link_end_hidden[:, 1:, :]  # b,l-1,h
        link_dot_prod_scores = torch.sum(link_dot_prod_scores, dim=-1)  # b,l-1
        link_dot_prod_scores = link_dot_prod_scores / hidden_size ** 0.5  # b,l-1
    else:
        # link score 普通计算方式 通过计算矩阵后取对角线 有大量非对角线的无用计算
        link_dot_prod_scores = torch.matmul(link_start_hidden, link_end_hidden.transpose(-1, -2))  # b,l,l
        link_dot_prod_scores = link_dot_prod_scores / hidden_size ** 0.5  # b,e,l,l
        link_dot_prod_scores = torch.diagonal(link_dot_prod_scores, offset=1, dim1=-2, dim2=-1)  # b,l-1

    return link_dot_prod_scores  # b,l-1
    # return torch.relu(link_dot_prod_scores)  # b,l-1


def calc_refined_mat_tensor(link_scores, pooling_type, temp=1):
    # link_scores [b,l-1]
    # span_ner_mat_tensor [b,l,l,e]
    if pooling_type == 'softmin':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='min', use_soft=True, temp=temp)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'min':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='min', use_soft=False)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'softmax':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='max', use_soft=True, temp=temp)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'max':
        mask_matrix = aggregate_mask_by_reduce(link_scores, mode='max', use_soft=False)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'mean':
        mask_matrix = aggregate_mask_by_cum(link_scores, mean=True)[..., None]  # b,l-1,l-1,1
    elif pooling_type == 'sum':
        mask_matrix = aggregate_mask_by_cum(link_scores, mean=False)[..., None]  # b,l-1,l-1,1
    else:
        raise NotImplementedError
    final_mask = torch.nn.functional.pad(mask_matrix, pad=(0, 0, 1, 0, 0, 1), mode="constant", value=0)  # b,l,l,1  # 长宽增加1对齐

    return final_mask


def aggregate_mask_by_reduce(tensor1, mode='max', use_soft=True, temp=1):
    """目前在用"""
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    if mode in ['max', 'min']:
        triu_mask = torch.triu(torch.ones([length, length]), diagonal=1).to(tensor1.device)[None, ...]  # 1,l,l
        """triu_mask
        [0., 1., 1.]
        [0., 0., 1.]
        [0., 0., 0.]
        """
        inv_triu_mask = torch.flip(triu_mask, dims=[-1])
        """inv_triu_mask
        [1., 1., 0.]
        [1., 0., 0.]
        [0., 0., 0.]
        """
        if mode == 'max':
            inv_triu_mask = inv_triu_mask * -1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [-inf., -inf., 3.]
            [-inf., 2., 2.]
            [1., 1., 1.]
            """

            if use_soft:
                cum_t = torch.logcumsumexp(cum_t / temp, dim=-2) * temp  # [b,l,l]
            else:
                cum_t, _ = torch.cummax(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote -inf
            [max0., max0., max3.]
            [max0+0., max2+0., max2+3.]
            [max1+0+0., max1+2+0., max1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [max0., max2., max2+3.]
            [max0., max0., max3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [0., max2., max2+3.]
            [0., 0., max3.]
            """


        elif mode == 'min':
            inv_triu_mask = inv_triu_mask * 1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [inf., inf., 3.]
            [inf., 2., 2.]
            [1., 1., 1.]
            """

            if use_soft:
                cum_t = torch.logcumsumexp(-cum_t / temp, dim=-2) * temp  # [b,l,l]
                cum_t = - cum_t
            else:
                cum_t, _ = torch.cummin(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote inf
            [min0., min0., min3.]
            [min0+0., min2+0., min2+3.]
            [min1+0+0., min1+2+0., min1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [min0., min2., min2+3.]
            [min0., min0., min3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [0., min2., min2+3.]
            [0., 0., min3.]
            """

    return cum_t


def aggregate_mask_by_cum(tensor1, mean=True):
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    # diag_mask = torch.diag_embed(torch.ones([length]), offset=0)  # [l,l]
    # diag_mask = diag_mask[None, ..., None]  # [1,l,l,1]
    # torch.diag_embed(tensor1, )

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    cum_t = torch.cumsum(cum_t, dim=-2)  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2+3.]
    [1., 1+2., 1+2+3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [1., 1+2., 1+2+3.]
    [0., 2., 2+3.]
    [0., 0., 3.]
    """
    sum_t = cum_t

    """构造相关mask矩阵"""
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ...]  # 1,l,l  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask

    if mean:
        # 求平均逻辑
        # 分母： 要除以来求平均
        # e.g. length=3
        heng = torch.arange(1, length + 1).to(tensor1.device)  # [1,2,3]
        heng = heng.unsqueeze(0).repeat((batch_size, 1))  # b,l
        heng = heng.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        """
        [1,2,3]
        [1,2,3]
        [1,2,3]
        """
        shu = torch.arange(0, length).to(tensor1.device)  # [0,1,2]
        shu = shu.unsqueeze(0).repeat((batch_size, 1))  # b,l
        shu = shu.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        shu = shu.transpose(1, 2)
        shu = - shu
        """
        [-0,-0,-0]
        [-1,-1,-1]
        [-2,-2,-2]
        """
        count = heng + shu  # 这里一开始竟然用了- --得正 日
        """
        [1,2,3]
        [0,1,2]
        [-1,0,1]  # 下三角会被mask掉不用管  Note:但是除以不能为0！
        """

        # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
        count = count * triu_mask + ignore_mask

        sum_t = sum_t / count

    # 再把下三角强制变为0
    sum_t = sum_t * triu_mask
    return sum_t
