import os, sys, copy, time, argparse, subprocess
from pathlib import Path
from typing import *
from tqdm import tqdm
# import ipdb
import torch
import numpy as np
from torch.utils.data import DataLoader
import logging, pprint
import datautils as utils
import modules
import ner_loader
from datasets import load_metric
from transformers import SchedulerType

torch.set_printoptions(linewidth=4000, sci_mode=False)  # How long line breaks, not print scientific notation
np.set_printoptions(linewidth=4000, suppress=True)  # How long line breaks, not print scientific notation


# from torch.utils.tensorboard import SummaryWriter


def add_label_token_bert(model: modules.MLM, loader, label_map):
    tokenizer = loader.datareader.tokenizer
    ###########  TODO:验证下面代码在fewnerd上是否兼容
    # 任务顺序
    task_order = [tid for tid, _ in sorted(loader.tid2entids.items(), key=lambda x: x[0])]
    # 根据任务顺序排序标签
    sorted_add_tokens = [label for tid in task_order for label in label_map.keys() if
                         label.startswith('I-' + loader.datareader.id2ent[tid])]
    ##########
    # sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)    # 将['I-MISC', 'I-PER', 'I-ORG', 'I-LOC']这4个词添加到词表vocab中，原始词表从28996变为29000
    num_tokens, _ = model.bert_layer.bert.embeddings.word_embeddings.weight.shape
    model.bert_layer.resize_token_embeddings(len(sorted_add_tokens)+num_tokens)  # 将bert模型的嵌入维度从原始的28996改为29000
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index>=num_tokens, (index, num_tokens, token)
            if isinstance(label_map[token], list):
                indexes = tokenizer.convert_tokens_to_ids(label_map[token])  # 得到'I-MISC'对应的那top6个词的词表索引
            else:
                indexes = tokenizer.convert_tokens_to_ids([label_map[token]])
            embed = model.bert_layer.bert.embeddings.word_embeddings.weight.data[indexes[0]]  # 得到23755索引(也就是'Palestinians'这个词的词嵌入权重矩阵，是一个768维的值

            # Calculate mean vector if there are multiple label words.
            for i in indexes[1:]:
                embed += model.bert_layer.bert.embeddings.word_embeddings.weight.data[i]  # 将其余5个词的权重矩阵和第一个词'Palestinians'的权重矩阵全部相加
            embed /= len(indexes)   # 求平均
            model.bert_layer.bert.embeddings.word_embeddings.weight.data[index] = embed   # 将这个平均值作为I-MISC这个词的词嵌入权重矩阵。其余三个(I-PER、I-ORG、I-LOC同理)

    return tokenizer


def add_label_token_roberta(model: modules.MLM, loader, label_map):
    tokenizer = loader.datareader.tokenizer
    sorted_add_tokens = sorted(list(label_map.keys()), key=lambda x: len(x), reverse=True)
    tokenizer.add_tokens(sorted_add_tokens)
    num_tokens, _ = model.roberta.embeddings.word_embeddings.weight.shape
    model.resize_token_embeddings(len(sorted_add_tokens)+num_tokens)
    for token in sorted_add_tokens:
        if token.startswith('B-') or token.startswith('I-'):  # 特殊字符
            index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
            if len(index)>1:
                raise RuntimeError(f"{token} wrong split: {index}")
            else:
                index = index[0]
            # assert index>=num_tokens, (index, num_tokens, token)
            if isinstance(label_map[token], list):
                indexes = tokenizer.convert_tokens_to_ids(label_map[token])
            else:
                indexes = tokenizer.convert_tokens_to_ids([label_map[token]])
            embed = model.roberta.embeddings.word_embeddings.weight.data[indexes[0]]

            # Calculate mean vector if there are multiple label words.
            for i in indexes[1:]:
                embed += model.roberta.embeddings.word_embeddings.weight.data[i]
            embed /= len(indexes)
            model.roberta.embeddings.word_embeddings.weight.data[index] = embed

    return tokenizer


def mlm_eval(model, test_dataloader, task_id, loader, info_str, mode='so_far', saving_exm_file=None, save_prob_dct=None):
    # torch.cuda.empty_cache()  # clean unactivate cuda memory
    model.eval()
    id2ent = loader.datareader.id2ent
    so_far_task_entids = sum([loader.tid2entids[tid] for tid in range(0, task_id + 1)], [])  # ent learned so far 截止当前任务学到的所有ent id
    curr_task_entids = loader.tid2entids[task_id]  # ent id learned in current task 当前任务学到的ent id
    offset1, offset2 = loader.tid2offset[task_id]  # ent position learned in current task 当前任务学到的ent位置
    so_far_task_ent = [id2ent[eid] for eid in so_far_task_entids]
    curr_task_ent = [id2ent[eid] for eid in curr_task_entids]

    f1_meaner = utils.F1_Meaner()
    # span_loss_meaner = utils.Meaner()
    mlm_loss_meaner = utils.Meaner()
    kl_loss_meaner = utils.Meaner()
    tmp_exm_lst = []
    probe_dct = {'batch_exm': []}
    start = time.time()
    token_list = []
    y_true = []
    y_pred = []
    iterator = tqdm(test_dataloader, ncols=300, dynamic_ncols=True)
    for i, inputs_dct in enumerate(iterator):  # iter steps
        seq_len = inputs_dct['ori_seq_len']
        batch_ner_exm = inputs_dct['batch_ner_exm']
        with torch.no_grad():
            if mode == 'so_far':  # calculate performance of ent so far
                # batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, task_id, mode='test')  # 计算截至当前任务的所有实体
                preds, refs, tokens, entlm_outputs_prob, batch_entlm_logits_list, f1, detail_f1 = model(inputs_dct, task_id, mode='test')  # 计算截至当前任务的所有实体
            if mode == 'curr':  # calculate performance of ent currently
                # batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, task_id, mode='train')  # 只要当前任务的实体
                preds, refs, tokens, entlm_outputs_prob, batch_entlm_logits_list, f1, detail_f1 = model(inputs_dct, task_id, mode='train')  # 只要当前任务的实体

        entlm_outputs_prob = [item.tolist() for item in entlm_outputs_prob]
        for exm, length, pred_prob, preds_tag in zip(batch_ner_exm, seq_len.tolist(), entlm_outputs_prob, preds):
            # ipdb.set_trace()
            tmp_exm = copy.deepcopy(exm)
            if mode == 'so_far':
                tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
            if mode == 'curr':
                tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)
            tmp_exm.pred_ent_dct = bio_to_span(preds_tag, pred_prob)
            # pred_ent_dct_entlm, _ = utils.NerExample.extract_entity_by_tags(preds_tag)
            # for k, v_lst in pred_ent_dct_entlm.items():
            #     for e in v_lst:
            #         e.append(1.)  # dummpy prob 1.
            # tmp_exm.pred_ent_dct = pred_ent_dct_entlm
            tmp_exm_lst.append(tmp_exm)

        iterator.set_description(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size}')
        # iterator.set_description(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size} | '
        #                          f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} '
        #                          f'Loss:{mlm_loss_meaner.v:.3f} {kl_loss_meaner.v:.3f}')

        # token_list.extend(tokens)
        # y_true.extend(refs)
        # y_pred.extend(preds)
        #
        # metric_mlm.add_batch(
        #     predictions=preds,
        #     references=refs,
        # )
    # eval_metric = compute_metrics()

        # batch_predict = batch_predict.detach()
        # batch_predict = torch.sigmoid(batch_predict).cpu()  # Not forget activate!
    print("验证完毕！解码用时(Decoding time): {}s".format(time.time() - start))
    if save_prob_dct:
        for span_tensor, l, exm in zip(model.batch_span_tensor, seq_len.tolist(), batch_ner_exm):
            _exm = exm.initial_new()
            span_tensor = span_tensor[:l, :l, :].cpu().numpy()
            _exm.span_tensor = span_tensor
            # probe_dct['batch_predict'].append(span_tensor)
            probe_dct['batch_exm'].append(_exm)

    # f1_meaner.add(*detail_f1)
    # f1_meaner.f1 = eval_metric['overall_f1']  # TODO 不要写死,不同任务取不同标签的f1
    # f1_meaner.rec = eval_metric['overall_recall']
    # f1_meaner.prec = eval_metric['overall_precision']
    # span_loss_meaner.add(span_loss.item())
    # mlm_loss_meaner.add(mlm_loss.item())
    # if kl_loss is not None and abs(kl_loss) > 1e-8:
    #     kl_loss_meaner.add(kl_loss.item())
        # if mode == 'curr':  # or train
        #     batch_predict = torch.nn.functional.pad(batch_predict, (offset1, 0), mode='constant', value=0.0)  # 在ent维度左边补0 右边不补

        # batch_predict_lst = torch.split(batch_predict, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

        # for exm, length, pred_prob in zip(batch_ner_exm, seq_len.tolist(), batch_predict_lst):
        #     # ipdb.set_trace()
        #     tmp_exm = copy.deepcopy(exm)
        #     if mode == 'so_far':
        #         tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
        #     if mode == 'curr':
        #         tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)
        #     tmp_exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(pred_prob.numpy(), length, id2ent)  # sigmoid
        #     tmp_exm_lst.append(tmp_exm)

    # iterator.set_description(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size} | '
    #                             f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} '
    #                             # f'Loss:{span_loss_meaner.v:.3f} {kl_loss_meaner.v:.3f}')
    #                             f'Loss:{mlm_loss_meaner.v:.3f} {kl_loss_meaner.v:.3f}')
    # eval_metric = compute_metrics()
    # tqdm.write(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size} | '
    # # logger.info(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size} | '
    #                             f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} '
    #                             # f'Loss:{span_loss_meaner.v:.3f} {kl_loss_meaner.v:.3f}')
    #                             f'Loss:{mlm_loss_meaner.v:.3f} {kl_loss_meaner.v:.3f}')


    m = {}  # to store metrics
    m['mip'], m['mir'], m['mif1'], m['map'], m['mar'], m['maf1'], detail_info_str, detail_stat = utils.NerExample.eval(
        tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=True, macro=True)
    logger.info(f'{info_str} Entity-Level Detailed Results:\n{detail_info_str}')
    m['detail_stat'] = detail_stat
    new_detail_stat = utils.metric_aggregater(detail_stat, {lst[0].split('-')[0]: lst for lst in loader.entity_task_lst})
    m['new_detail_stat'] = new_detail_stat
    logger.info('Task-Level Aggregated Entity Results:')
    for ent, v in new_detail_stat.items():
        logger.info(f'{ent}:\t\tMicroF1:{v["mif1"]:.3%} MacroF1:{v["maf1"]:.3%}')
    logger.info(f'{info_str} Overall:\t\tMicroF1:{m["mif1"]:.3%} MacroF1:{m["maf1"]:.3%}')
    final_macro_f1_over_task_level = mean([v["mif1"] for ent, v in new_detail_stat.items()])
    logger.info(f'{info_str} Final MacroF1 over Task-Level:{final_macro_f1_over_task_level:.3%}')
    utils.NerExample.save_to_jsonl(tmp_exm_lst, f'{saving_exm_file}') if saving_exm_file is not None else None
    torch.save(probe_dct, save_prob_dct) if save_prob_dct else None
    # 打印结果
    # m = {}  # to store metrics
    # # raw_f1 = f1_meaner.f1  # raw means use_flat_pred_ent_dct=False, i,e, not remove the overlapped
    # m['raw_mip'], m['raw_mir'], m['raw_mif1'], m['raw_map'], m['raw_mar'], m['raw_maf1'], _, _ = utils.NerExample.eval(
    #     tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=False, macro=True)  # mi=micro; ma=macro
    # m['mip'], m['mir'], m['mif1'], m['map'], m['mar'], m['maf1'], detail_info_str, detail_stat = utils.NerExample.eval(
    #     tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=True, macro=True)
    # logger.info(f'{info_str} Entity-Level Detailed Results:\n{detail_info_str}')
    # m['detail_stat'] = detail_stat
    # new_detail_stat = utils.metric_aggregater(detail_stat, {lst[0].split('-')[0]: lst for lst in loader.entity_task_lst})
    # m['new_detail_stat'] = new_detail_stat
    # logger.info('Task-Level Aggregated Entity Results:')
    # for ent, v in new_detail_stat.items():
    #     logger.info(f'{ent}:\t\tMicroF1:{v["mif1"]:.3%} MacroF1:{v["maf1"]:.3%}')
    # # logger.info(f'***{info_str} | raw_mif1:{m["raw_mif1"]:.5f} raw_maf1:{m["raw_maf1"]:.5f} mif1:{m["mif1"]:.5f} maf1:{m["maf1"]:.5f}')
    # logger.info(f'{info_str} Overall:\t\tMicroF1:{m["mif1"]:.3%} MacroF1:{m["maf1"]:.3%}')
    # final_macro_f1_over_task_level = mean([v["mif1"] for ent, v in new_detail_stat.items()])
    # logger.info(f'{info_str} Final MacroF1 over Task-Level:{final_macro_f1_over_task_level:.3%}')
    # utils.NerExample.save_to_jsonl(tmp_exm_lst, f'{saving_exm_file}', only_pred_str=False, flat_pred_ent=True) if saving_exm_file is not None else None
    # torch.save(probe_dct, save_prob_dct) if save_prob_dct else None
    # return m
    return m


def bio_to_span(predictions, probs):
    spans = {}
    current_label = None
    current_start = None

    for i, (label, prob) in enumerate(zip(predictions, probs)):
        if label.startswith('B-'):
            if current_label is not None:
                if current_label not in spans:
                    spans[current_label] = []
                spans[current_label].append([current_start, i, probs[current_start]])
            current_label = label[2:]  # 当前标签类型
            current_start = i
        elif label.startswith('I-'):
            continue
        elif label == 'O' and current_label is not None:
            if current_label not in spans:
                spans[current_label] = []
            spans[current_label].append([current_start, i, probs[current_start]])
            current_label = None

    # Handle the case where the last token is part of a span
    if current_label is not None:
        if current_label not in spans:
            spans[current_label] = []
        spans[current_label].append([current_start, len(predictions), probs[current_start]])

    # Convert span format to the desired output
    result = {key: [[start, end, prob] for start, end, prob in value] for key, value in spans.items()}
    return result


def softmax(array, dim=-1):
    # exps = np.exp(array - np.max(array))
    exps = np.exp(array)
    return exps / np.sum(exps, axis=dim, keepdims=True)


def spankl_eval(model, test_dataloader, task_id, loader, info_str, mode='so_far', saving_exm_file=None, save_prob_dct=None):
    # torch.cuda.empty_cache()  # clean unactivate cuda memory
    model.eval()
    id2ent = loader.datareader.id2ent
    so_far_task_entids = sum([loader.tid2entids[tid] for tid in range(0, task_id + 1)], [])  # ent learned so far 截止当前任务学到的所有ent id
    curr_task_entids = loader.tid2entids[task_id]  # ent id learned in current task 当前任务学到的ent id
    offset1, offset2 = loader.tid2offset[task_id]  # ent position learned in current task 当前任务学到的ent位置
    so_far_task_ent = [id2ent[eid] for eid in so_far_task_entids]
    curr_task_ent = [id2ent[eid] for eid in curr_task_entids]

    f1_meaner = utils.F1_Meaner()
    span_loss_meaner = utils.Meaner()
    kl_loss_meaner = utils.Meaner()
    tmp_exm_lst = []
    probe_dct = {'batch_exm': []}
    iterator = tqdm(test_dataloader, ncols=300, dynamic_ncols=True)
    for i, inputs_dct in enumerate(iterator):  # iter steps
        seq_len = inputs_dct['ori_seq_len']
        batch_ner_exm = inputs_dct['batch_ner_exm']
        with torch.no_grad():
            if mode == 'so_far':  # calculate performance of ent so far
                batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, task_id, mode='test')  # 计算截至当前任务的所有实体
            if mode == 'curr':  # calculate performance of ent currently
                batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, task_id, mode='train')  # 只要当前任务的实体
        batch_predict = batch_predict.detach()
        batch_predict = torch.sigmoid(batch_predict).cpu()  # Not forget activate!   sigmoid将数值映射到 0 到 1 之间

        if save_prob_dct:
            for span_tensor, l, exm in zip(model.batch_span_tensor, seq_len.tolist(), batch_ner_exm):
                _exm = exm.initial_new()
                span_tensor = span_tensor[:l, :l, :].cpu().numpy()
                _exm.span_tensor = span_tensor
                # probe_dct['batch_predict'].append(span_tensor)
                probe_dct['batch_exm'].append(_exm)

        f1_meaner.add(*detail_f1)   # detail_f1包含(num_gold, num_pred, tp.item())
        span_loss_meaner.add(span_loss.item())
        if kl_loss is not None:
            kl_loss_meaner.add(kl_loss.item())
        if mode == 'curr':  # or train
            batch_predict = torch.nn.functional.pad(batch_predict, (offset1, 0), mode='constant', value=0.0)  # 在ent维度左边补0 右边不补

        batch_predict_lst = torch.split(batch_predict, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]
        # 下面for循环主要目的是将预测的.pred_ent_dct替换原始数据tmp_exm中
        for exm, length, pred_prob in zip(batch_ner_exm, seq_len.tolist(), batch_predict_lst):
            # ipdb.set_trace()
            tmp_exm = copy.deepcopy(exm)
            if mode == 'so_far':
                tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
            if mode == 'curr':
                tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)
            tmp_exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(pred_prob.numpy(), length, id2ent)  # sigmoid
            tmp_exm_lst.append(tmp_exm)

        iterator.set_description(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size} | '
                                 f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} '
                                 f'Loss:{span_loss_meaner.v:.3f} {kl_loss_meaner.v:.3f}')

    m = {}  # to store metrics
    # raw_f1 = f1_meaner.f1  # raw means use_flat_pred_ent_dct=False, i,e, not remove the overlapped
    m['raw_mip'], m['raw_mir'], m['raw_mif1'], m['raw_map'], m['raw_mar'], m['raw_maf1'], _, _ = utils.NerExample.eval(
        tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=False, macro=True)  # mi=micro; ma=macro
    m['mip'], m['mir'], m['mif1'], m['map'], m['mar'], m['maf1'], detail_info_str, detail_stat = utils.NerExample.eval(
        tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=True, macro=True)
    logger.info(f'{info_str} Entity-Level Detailed Results:\n{detail_info_str}')
    m['detail_stat'] = detail_stat
    new_detail_stat = utils.metric_aggregater(detail_stat, {lst[0].split('-')[0]: lst for lst in loader.entity_task_lst})
    m['new_detail_stat'] = new_detail_stat
    logger.info('Task-Level Aggregated Entity Results:')
    for ent, v in new_detail_stat.items():
        logger.info(f'{ent}:\t\tMicroF1:{v["mif1"]:.3%} MacroF1:{v["maf1"]:.3%}')
    # logger.info(f'***{info_str} | raw_mif1:{m["raw_mif1"]:.5f} raw_maf1:{m["raw_maf1"]:.5f} mif1:{m["mif1"]:.5f} maf1:{m["maf1"]:.5f}')
    logger.info(f'{info_str} Overall:\t\tMicroF1:{m["mif1"]:.3%} MacroF1:{m["maf1"]:.3%}')
    final_macro_f1_over_task_level = mean([v["mif1"] for ent, v in new_detail_stat.items()])
    logger.info(f'{info_str} Final MacroF1 over Task-Level:{final_macro_f1_over_task_level:.3%}')
    utils.NerExample.save_to_jsonl(tmp_exm_lst, f'{saving_exm_file}', only_pred_str=False, flat_pred_ent=True) if saving_exm_file is not None else None
    torch.save(probe_dct, save_prob_dct) if save_prob_dct else None
    return m


def baseline_ext_eval(model: modules.BaselineExtend, test_dataloader, task_id, loader, info_str, mode='so_far',
                      saving_exm_file=None, save_prob_dct=None):
    model.eval()
    id2tag = loader.datareader.id2tag
    id2ent = loader.datareader.id2ent
    so_far_task_entids = sum([loader.tid2entids[tid] for tid in range(0, task_id + 1)], [])  # ent learned so far 截止当前任务学到的所有ent id
    curr_task_entids = loader.tid2entids[task_id]
    so_far_task_ent = [id2ent[eid] for eid in so_far_task_entids]
    curr_task_ent = [id2ent[eid] for eid in curr_task_entids]
    tmp_exm_lst = []
    probe_dct = {'batch_exm': []}
    iterator = tqdm(test_dataloader, ncols=300, dynamic_ncols=True)
    for i, inputs_dct in enumerate(iterator):  # iter steps
        batch_ner_exm = inputs_dct['batch_ner_exm']
        with torch.no_grad():
            ent_output = model.encode(inputs_dct)
        seq_len = inputs_dct['ori_seq_len']

        if save_prob_dct:  # save logit for Value Probe
            for tag_tensor, l, exm in zip(model.batch_tag_tensor, seq_len, batch_ner_exm):
                _exm = exm.initial_new()
                tag_tensor = tag_tensor[:l, :].cpu().numpy()
                _exm.tag_tensor = tag_tensor
                probe_dct['batch_exm'].append(_exm)

        # if task_id == 1: ipdb.set_trace()
        use_viterbi_decode = True  # ExtendNER paper use viterbi decode, i,e, a CRF mask.
        if use_viterbi_decode:  # using viterbi_decode with a prior valid transition mask
            decode_ids = model.decode(ent_output, seq_len, task_id)
            for exm, decode_ids_ in zip(batch_ner_exm, decode_ids):
                tag_lst = [id2tag[tag_id] for tag_id in decode_ids_]
                pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                for k, v_lst in pred_ent_dct.items():
                    for e in v_lst:
                        e.append(1.)  # dummpy prob 1.
                if mode == 'curr':  # only consider current ent 只看当前实体
                    pred_ent_dct = {k: v_lst for k, v_lst in pred_ent_dct.items() if k in curr_task_ent}
                tmp_exm = copy.deepcopy(exm)
                tmp_exm.pred_ent_dct = pred_ent_dct
                if mode == 'curr':
                    tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)
                else:  # test
                    tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
                tmp_exm_lst.append(tmp_exm)
        else:
            ofs, ofe = model.task_offset_lst[task_id]  # offset in current task 当前任务的offset
            task_ent_output = ent_output[:, :, :ofe]
            task_ent_output_tag_id = task_ent_output.argmax(-1)  # [b,l]
            task_ent_output_tag_id = task_ent_output_tag_id.cpu().detach().numpy()
            seq_len = seq_len.tolist()
            for exm, prob, length in zip(batch_ner_exm, task_ent_output_tag_id, seq_len):
                tag_lst = [id2tag[tag_id] for tag_id in prob[:length]]
                pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                for k, v_lst in pred_ent_dct.items():
                    for e in v_lst:
                        e.append(1.)  # dummpy prob 1.
                if mode == 'curr':  # only consider current ent 只看当前实体
                    pred_ent_dct = {k: v_lst for k, v_lst in pred_ent_dct.items() if k in curr_task_ent}
                tmp_exm = copy.deepcopy(exm)
                tmp_exm.pred_ent_dct = pred_ent_dct
                if mode == 'curr':
                    tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)
                else:  # test
                    tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
                tmp_exm_lst.append(tmp_exm)
                # if 'NORP' in tmp_exm.pred_ent_dct: ipdb.set_trace()

        iterator.set_description(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size}')

    m = {}  # to store metrics
    m['mip'], m['mir'], m['mif1'], m['map'], m['mar'], m['maf1'], detail_info_str, detail_stat = utils.NerExample.eval(
        tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=True, macro=True)
    logger.info(f'{info_str} Entity-Level Detailed Results:\n{detail_info_str}')
    m['detail_stat'] = detail_stat
    new_detail_stat = utils.metric_aggregater(detail_stat, {lst[0].split('-')[0]: lst for lst in loader.entity_task_lst})
    m['new_detail_stat'] = new_detail_stat
    logger.info('Task-Level Aggregated Entity Results:')
    for ent, v in new_detail_stat.items():
        logger.info(f'{ent}:\t\tMicroF1:{v["mif1"]:.3%} MacroF1:{v["maf1"]:.3%}')
    logger.info(f'{info_str} Overall:\t\tMicroF1:{m["mif1"]:.3%} MacroF1:{m["maf1"]:.3%}')
    final_macro_f1_over_task_level = mean([v["mif1"] for ent, v in new_detail_stat.items()])
    logger.info(f'{info_str} Final MacroF1 over Task-Level:{final_macro_f1_over_task_level:.3%}')
    utils.NerExample.save_to_jsonl(tmp_exm_lst, f'{saving_exm_file}') if saving_exm_file is not None else None
    torch.save(probe_dct, save_prob_dct) if save_prob_dct else None
    return m


def baseline_add_eval(model: modules.BaselineAdd, test_dataloader, task_id, loader, info_str, mode='so_far',
                      saving_exm_file=None, save_prob_dct=None):
    model.eval()
    id2tag = loader.datareader.id2tag
    id2ent = loader.datareader.id2ent
    so_far_task_entids = sum([loader.tid2entids[tid] for tid in range(0, task_id + 1)], [])  # ent learned so far 截止当前任务学到的所有ent id
    curr_task_entids = loader.tid2entids[task_id]
    so_far_task_ent = [id2ent[eid] for eid in so_far_task_entids]
    curr_task_ent = [id2ent[eid] for eid in curr_task_entids]
    offset_split = model.offset_split
    tid_id2tag_lst = model.tid_id2tag_lst
    tmp_exm_lst = []
    probe_dct = {'batch_exm': []}
    iterator = tqdm(test_dataloader, ncols=300, dynamic_ncols=True)
    for i, inputs_dct in enumerate(iterator):  # iter step
        batch_ner_exm = inputs_dct['batch_ner_exm']
        with torch.no_grad():
            ent_output1 = model.encode(inputs_dct)
        ent_output = ent_output1.cpu().numpy()
        seq_len = inputs_dct['ori_seq_len'].tolist()

        if save_prob_dct:  # save logit for Value Probe
            for tag_tensor, l, exm in zip(model.batch_tag_tensor, seq_len, batch_ner_exm):
                _exm = exm.initial_new()
                tag_tensor = tag_tensor[:l, :].cpu().numpy()
                _exm.tag_tensor = tag_tensor
                probe_dct['batch_exm'].append(_exm)

        decode_mode = 'addner'  # adder paper method. have two implementation ways

        # decode_mode = 'addner_viterbi'  # for each head use viterbi_decode with transition mask
        # 效果与addner也差不多 因为只有1个实体的情况下加crf_mask没有用，全部都是允许的通路

        # decode_mode = 'extendner'  # like extendNER, use viterbi for all head

        if decode_mode == 'addner':  # 各个头抽出来后启发式合并 heuristically merge each head's resutl as AddNER paper
            for exm, tag_logits, l in zip(batch_ner_exm, ent_output, seq_len):
                tag_logits = tag_logits[:l, :]
                tmp_exm = copy.deepcopy(exm)
                if mode == 'curr':
                    tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)  # 只保留当前任务的实体黄金标签
                if mode == 'so_far':
                    tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)

                task_tag_logits_lst = np.split(tag_logits, offset_split, axis=-1)[:task_id + 1]
                task_tag_probs_lst = [softmax(e) for e in task_tag_logits_lst]  # all probs
                task_tag_prob_lst = [e.max(-1).tolist() for e in task_tag_probs_lst]  # max prob
                task_tag_ids_lst = [e.argmax(-1) for e in task_tag_logits_lst]  # tag id
                task_tag_str_lst = [[tid_id2tag_lst[tid][tag_id] for tag_id in e] for tid, e in enumerate(task_tag_ids_lst)]  # tag str

                if mode == 'curr':
                    task_tag_logits_lst = task_tag_logits_lst[-1:]
                    task_tag_probs_lst = task_tag_probs_lst[-1:]
                    task_tag_prob_lst = task_tag_prob_lst[-1:]
                    task_tag_ids_lst = task_tag_ids_lst[-1:]
                    task_tag_str_lst = task_tag_str_lst[-1:]

                """ my decode strategy """
                # pred_ent_dct = {}
                # for task_tag_str, task_tag_prob in zip(task_tag_str_lst, task_tag_prob_lst):
                #     task_pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(task_tag_str)
                #     for k, v_lst in task_pred_ent_dct.items():
                #         for v in v_lst:
                #             s, e = v[:2]
                #             prob = np.mean(task_tag_prob[s:e])
                #             v.append(prob)
                #             # v.append(1.)  # 假设概率为1
                #     pred_ent_dct.update(task_pred_ent_dct)
                # tmp_exm.pred_ent_dct = pred_ent_dct
                # tmp_exm.pred_ent_dct = tmp_exm.get_flat_pred_ent_dct()
                # tmp_exm_lst.append(tmp_exm)
                """ end """

                """ addner paper decode strategy """  # We found it final equal to [my decode strategy]
                combine_tag_lst = []
                for tok_idx, all_task_tags in enumerate(zip(*task_tag_str_lst)):
                    # tok_idx:  token position
                    # all_task_tags: the token's predicted tag of all tasks
                    all_task_tags = list(all_task_tags)
                    for tidx, tag in enumerate(all_task_tags):  # iter among tasks
                        if tag[:1] == 'I':
                            if combine_tag_lst and combine_tag_lst[-1] in [f'B-{tag[2:]}', f'I-{tag[2:]}']:
                                combine_tag_lst.append(tag)
                                break
                            else:
                                all_task_tags[tidx] = 'O'
                    if len(combine_tag_lst) == (tok_idx + 1):  # already consider adding I 已经加了I
                        continue
                    tids = [tidx for tidx, tag in enumerate(all_task_tags) if tag[:1] == 'B']
                    if not tids:
                        combine_tag_lst.append('O')
                    else:
                        bscores = [(task_tag_prob_lst[tidx][tok_idx], tidx) for tidx in tids]
                        max_tidx = sorted(bscores, key=lambda e: e[0])[-1][1]
                        combine_tag_lst.append(all_task_tags[max_tidx])
                # print(combine_tag_lst)
                pred_ent_dct_addner_paper, _ = utils.NerExample.extract_entity_by_tags(combine_tag_lst)   # 将BIO转为span
                for k, v_lst in pred_ent_dct_addner_paper.items():
                    for e in v_lst:
                        e.append(1.)  # dummpy prob 1.
                tmp_exm.pred_ent_dct = pred_ent_dct_addner_paper
                tmp_exm_lst.append(tmp_exm)
                """ end """

        elif decode_mode == 'addner_viterbi':  # for each task head use viterbi_decode with transition mask
            tmp_batch_ner_exm = copy.deepcopy(batch_ner_exm)
            for exm in tmp_batch_ner_exm:
                if mode == 'curr':
                    exm.remove_ent_by_type(curr_task_ent, input_keep=True)
                if mode == 'so_far':
                    exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
            for exm in tmp_batch_ner_exm:
                exm.pred_ent_dct = {}
            if mode == 'curr':
                decode_ids = model.decode_one_task(ent_output1, inputs_dct['ori_seq_len'], task_id)
                for exm, decode_ids_ in zip(tmp_batch_ner_exm, decode_ids):
                    tag_lst = [tid_id2tag_lst[task_id][tag_id] for tag_id in decode_ids_]
                    flat_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                    exm.pred_ent_dct.update(flat_ent_dct)
            else:
                for tid in range(task_id + 1):
                    decode_ids = model.decode_one_task(ent_output1, inputs_dct['ori_seq_len'], tid)
                    for exm, decode_ids_ in zip(tmp_batch_ner_exm, decode_ids):
                        tag_lst = [tid_id2tag_lst[tid][tag_id] for tag_id in decode_ids_]
                        flat_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                        exm.pred_ent_dct.update(flat_ent_dct)
            for exm in tmp_batch_ner_exm:
                # combine result of all head according to probs 根据概率合并所有头的结果
                exm.pred_ent_dct = utils.NerExample.Flat_ent_dct_by_prob(exm.pred_ent_dct, len(exm.char_lst), set_prob=1.)
            tmp_exm_lst.extend(tmp_batch_ner_exm)

        elif decode_mode == 'extendner':  # like Extend NER, use viterbi_decode with transition mask for all head once
            decode_ids = model.decode(ent_output1, inputs_dct['ori_seq_len'], task_id)
            for exm, decode_ids_ in zip(inputs_dct['batch_ner_exm'], decode_ids):
                tag_lst = [id2tag[tag_id] for tag_id in decode_ids_]
                pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                for k, v_lst in pred_ent_dct.items():
                    for e in v_lst:
                        e.append(1.)  # dummpy prob 1.
                if mode == 'curr':  # only consider current ent 只看当前实体
                    pred_ent_dct = {k: v_lst for k, v_lst in pred_ent_dct.items() if k in curr_task_ent}
                tmp_exm = copy.deepcopy(exm)
                tmp_exm.pred_ent_dct = pred_ent_dct
                if mode == 'curr':
                    tmp_exm.remove_ent_by_type(curr_task_ent, input_keep=True)
                if mode == 'so_far':
                    tmp_exm.remove_ent_by_type(so_far_task_ent, input_keep=True)
                tmp_exm_lst.append(tmp_exm)

        iterator.set_description(f'Task{task_id} {info_str}[{mode}] Step{i} | BS:{test_dataloader.batch_size}')

    m = {}  # to store metrics
    m['mip'], m['mir'], m['mif1'], m['map'], m['mar'], m['maf1'], detail_info_str, detail_stat = utils.NerExample.eval(
        tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=True, macro=True)
    logger.info(f'{info_str} Entity-Level Detailed Results:\n{detail_info_str}')
    m['detail_stat'] = detail_stat
    new_detail_stat = utils.metric_aggregater(detail_stat, {lst[0].split('-')[0]: lst for lst in loader.entity_task_lst})
    m['new_detail_stat'] = new_detail_stat
    logger.info('Task-Level Aggregated Entity Results:')
    for ent, v in new_detail_stat.items():
        logger.info(f'{ent}:\t\tMicroF1:{v["mif1"]:.3%} MacroF1:{v["maf1"]:.3%}')
    logger.info(f'{info_str} Overall:\t\tMicroF1:{m["mif1"]:.3%} MacroF1:{m["maf1"]:.3%}')
    final_macro_f1_over_task_level = mean([v["mif1"] for ent, v in new_detail_stat.items()])
    logger.info(f'{info_str} Final MacroF1 over Task-Level:{final_macro_f1_over_task_level:.3%}')
    utils.NerExample.save_to_jsonl(tmp_exm_lst, f'{saving_exm_file}') if saving_exm_file is not None else None
    torch.save(probe_dct, save_prob_dct) if save_prob_dct else None
    return m


def find_e(target, lst, end=1):
    [idx for idx, e in enumerate(lst) if target == e[:1]]


def mean(lst):
    return sum(lst) / len(lst)

def simply_print_cl_metric(model_ckpt):
    """ load the recorded overview_metric.json data to compute CL metrics used in the paper
        note that Few-NERD asks model to learn multiple entity types per task, and we use the microF1 over the entity types inner a task as this task's metric.
        For every step in CL, we use macroF1 over all the learned tasks as the final metric (refer to section Metrics), for example:
        Step1 is Micro(Task1)
        Step2 is Macro( Micro(Task1), Micro(Task2))
        Step3 is Macro( Micro(Task1), Micro(Task2), Micro(Task3))
    """
    from print_cl_metric import print_cl_metric
    metrics_json = utils.load_json(f'{model_ckpt}/overview_metric.json')
    model_ckpt = os.path.split(str(model_ckpt))[-1]
    test_metric = metrics_json['test_metric']
    filter_test_metric = metrics_json['filter_test_metric'] if 'filter_test_metric' in metrics_json else None
    perm = model_ckpt.split('_')[-1]
    print_cl_metric(metrics_json, model_ckpt, test_metric, perm, filter_test_metric=filter_test_metric, print_repr=False, print_detail=False)


def baseline_ner(model, loader, args, model_type, learn_mode='cl'):
    if model_type == 'ext':
        test_fn = baseline_ext_eval
    else:
        test_fn = baseline_add_eval

    test_dataloader_All = loader.get_task_dataloader(mode='test')
    test_tasks_dataloaders_Filter = loader.test_tasks_dataloaders_filtered

    start_taskid = 0
    start_model = ''
    end_taskid = 0

    metrics = {'test_metric': {},
               'dev_metric': {},
               'train_metric': {},
               'task_best_dev_epo': [-1] * loader.num_tasks,
               'filter_test_metric': {}
               }
    if start_model and os.path.exists(args.ckpt_dir / f'{start_model}/overview_metric.json'):
        metrics = utils.load_json(args.ckpt_dir / f'{start_model}/overview_metric.json')

    for task_id in range(loader.num_tasks):
        if start_model and start_taskid and task_id < start_taskid: continue
        if end_taskid and task_id >= end_taskid: break
        logger.info(utils.header_format(f'task {task_id} train', sep='='))
        metrics['test_metric'][task_id] = {}
        metrics['dev_metric'][task_id] = {}
        metrics['train_metric'][task_id] = {}
        metrics['filter_test_metric'][task_id] = {}


        if learn_mode == 'cl':
            train_dataloader = loader.get_task_dataloader(mode='train', tid=task_id)
            dev_dataloader = loader.get_task_dataloader(mode='dev', tid=task_id)
        if learn_mode == 'non_cl':
            train_dataloader = loader.so_far_train_tasks_dataloaders[task_id]
            dev_dataloader = loader.so_far_dev_tasks_dataloaders[task_id]

        # load possible last task model (the last model or the best model (use dev))
        if task_id > 0:
            if start_model and start_taskid and task_id == start_taskid:
                model.load_model(args.ckpt_dir / f'{start_model}/task_{start_taskid - 1}_model.pt',
                                 info=f'load existing start model:{start_model} task:{start_taskid - 1} success!')
            else:
                model.load_model(args.curr_ckpt_dir / f'task_{task_id - 1}_model.pt',
                                 info=f'load prev task model: task_{task_id - 1}_model.pt. last best_dev_epo: {metrics["task_best_dev_epo"][task_id - 1]}')

        # distill the knowledge from the previous model
        if learn_mode == 'cl' and args.use_distill and task_id > 0:
            last_task_id = task_id - 1
            model.eval()
            # distill Train set
            iterator = tqdm(train_dataloader, dynamic_ncols=True)
            [delattr(exm, 'distilled_task_ent_output') for exm in train_dataloader.dataset.instances if hasattr(exm, 'distilled_task_ent_output')]
            for i, inputs_dct in enumerate(iterator):  # iter step
                seq_len = inputs_dct['ori_seq_len']
                batch_ner_exm = inputs_dct['batch_ner_exm']
                ent_output = model.encode(inputs_dct)   # (bs, len, 18)其中18代表分类器输出维度
                ofs, ofe = model.task_offset_lst[last_task_id]  # offset of current task 当前任务的offset
                # task_ent_output_prob = ent_output[:,:,:ofe].softmax(-1)
                task_ent_output = ent_output[:, :, :ofe]
                # task_ent_output_tag_id = task_ent_output.argmax(-1)  # [b,l]
                task_ent_output = task_ent_output.cpu().detach().numpy()  # [b,l,ent]

                for exm, out, length in zip(batch_ner_exm, task_ent_output, seq_len):
                    # exm.distilled_task_ent_output = np.zeros(length, )
                    exm.distilled_task_ent_output = out[:length, :]  # [l,ent]
                iterator.set_description(f'Task{task_id} Distilling Train set Step{i}')

        if args.use_best_dev:
            best_dev_f1 = -1.
        model.init_opt()
        model.init_lrs(num_step_per_epo=len(train_dataloader), epo=args.num_epochs, num_warmup_steps=args.warmup_step)

        step_in_task = 0
        for ep in range(args.num_epochs):
            model.train()
            logger.info(utils.header_format(f'task {task_id} train epo {ep}', sep='='))
            iterator = tqdm(train_dataloader, ncols=300, dynamic_ncols=True)
            for i, inputs_dct in enumerate(iterator):  # iter steps
                step_in_task += 1
                if learn_mode == 'cl':
                    loss, ce_loss, kl_loss = model.runloss(inputs_dct, task_id)
                if learn_mode == 'non_cl':
                    loss, ce_loss, kl_loss = model.run_loss_non_cl(inputs_dct, task_id)
                iterator.set_description(
                    f'Task{task_id} Train Ep {ep}/{args.num_epochs} Step{i} | '
                    f'Loss:{loss:.3f} {ce_loss:.3f} {kl_loss:.3f} | '
                    f'BS:{train_dataloader.batch_size} LR:{model.curr_lr:.6f} '
                    f'gnorm:{model.total_norm:.3f} gclip:{model.grad_clip}'
                )
            # evalate
            utils.save_args_to_json_file(args, f'{args.curr_ckpt_dir}/args.json')
            if args.use_best_dev:
                if learn_mode == 'cl':
                    m = test_fn(model, dev_dataloader, task_id, loader, 'Dev  ', mode='curr')
                if learn_mode == 'non_cl':
                    m = test_fn(model, dev_dataloader, task_id, loader, 'Dev  ', mode='so_far')
                f1 = m['maf1']
                metrics['dev_metric'][task_id][ep] = f1
                if f1 > best_dev_f1:
                    best_dev_f1 = f1
                    metrics["task_best_dev_epo"][task_id] = ep
                    model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save_best_dev task_{task_id}_model.pt in epo: {ep} ')
            else:
                model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save_model.pt in task: {task_id} epo: {ep} ')

        if args.use_best_dev:
            model.load_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'loaded best dev model to test! task_best_dev_epo: {metrics["task_best_dev_epo"][task_id]}')  # obtain best model
        m = test_fn(model, test_dataloader_All, task_id, loader, info_str='Test All', mode='so_far',
                    saving_exm_file=f'{args.curr_ckpt_dir}/testFilter_exm_lst{task_id}.jsonl')
        metrics['test_metric'][task_id][ep] = m['detail_stat']
        metrics['test_metric'][task_id][ep]['micro_f1'] = m['mif1']
        metrics['test_metric'][task_id][ep]['macro_f1'] = m['maf1']
        if args.also_Test_Filter:
            m = test_fn(model, test_tasks_dataloaders_Filter[task_id], task_id, loader, info_str='Test Filter', mode='so_far',
                        saving_exm_file=f'{args.curr_ckpt_dir}/testFilter_exm_lst{task_id}.jsonl')
            metrics['filter_test_metric'][task_id][ep] = m['detail_stat']
            metrics['filter_test_metric'][task_id][ep]['micro_f1'] = m['mif1']
            metrics['filter_test_metric'][task_id][ep]['macro_f1'] = m['maf1']
        utils.save_json(metrics, f'{args.curr_ckpt_dir}/overview_metric.json')
        simply_print_cl_metric(args.curr_ckpt_dir)

    return None


def spankl_ner(model: modules.SpanKL, loader: ner_loader, args, learn_mode='cl'):
    test_dataloader_All: DataLoader = loader.get_task_dataloader(mode='test')
    test_tasks_dataloaders_Filter: List[DataLoader] = loader.test_tasks_dataloaders_filtered
    id2ent = loader.datareader.id2ent
    # tb_writer = SummaryWriter(f'./runs/{args.corpus}-{args.time_series}')
    args.test_per_epo = False
    args.use_finetune = False

    start_taskid = 0
    start_model = ''
    end_taskid = 0
    # start_taskid = 5
    # start_model = 'onto-0-2022-07-21_10-02-57-1824-spankl_split_perm0'
    # end_taskid = 1

    metrics = {'test_metric': {},
               'dev_metric': {},
               'train_metric': {},
               'task_best_dev_epo': [-1] * loader.num_tasks,
               'filter_test_metric': {}
               }
    if start_model and os.path.exists(args.ckpt_dir / f'{start_model}/overview_metric.json'):
        metrics = utils.load_json(args.ckpt_dir / f'{start_model}/overview_metric.json')

    for task_id in range(loader.num_tasks):
        if start_model and start_taskid and task_id < start_taskid: continue
        if end_taskid and task_id >= end_taskid: break
        logger.info(utils.header_format(f'task {task_id} train', sep='='))
        metrics['test_metric'][task_id] = {}
        metrics['dev_metric'][task_id] = {}
        metrics['train_metric'][task_id] = {}
        metrics['filter_test_metric'][task_id] = {}

        if learn_mode == 'cl':
            train_dataloader = loader.get_task_dataloader(mode='train', tid=task_id)   # 只包含task_id=0任务的train_dataloader
            dev_dataloader = loader.get_task_dataloader(mode='dev', tid=task_id)
        if learn_mode == 'non_cl':
            train_dataloader = loader.so_far_train_tasks_dataloaders[task_id]
            dev_dataloader = loader.so_far_dev_tasks_dataloaders[task_id]

        # load possible last task model (the last model or the best model (use dev))
        if task_id > 0:
            if start_model and start_taskid and task_id == start_taskid:
                model.load_model(args.ckpt_dir / f'{start_model}/task_{start_taskid - 1}_model.pt',
                                 info=f'load existing start model:{start_model} task:{start_taskid - 1} success!')
            else:
                model.load_model(args.curr_ckpt_dir / f'task_{task_id - 1}_model.pt',
                                 info=f'load prev task model: task_{task_id - 1}_model.pt. last best_dev_epo: {metrics["task_best_dev_epo"][task_id - 1]}')

        # distill the knowledge from the previous model
        if learn_mode == 'cl' and args.use_distill and task_id > 0:
            f1_meaner = utils.F1_Meaner()
            last_task_id = task_id - 1
            model.eval()
            # distill Train set
            iterator = tqdm(train_dataloader, dynamic_ncols=True)
            for i, inputs_dct in enumerate(iterator):  # iter steps
                # inputs_dct.pop('span_ner_tgt_lst')
                inputs_dct.pop('batch_span_tgt_lst_distilled')  # no need to return kl_loss
                seq_len = inputs_dct['ori_seq_len']
                with torch.no_grad():  #
                    batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, last_task_id, mode='test')
                batch_predict = batch_predict.detach().cpu()  # (644,1)
                f1_meaner.add(*detail_f1)
                batch_predict_lst = torch.split(batch_predict, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]
                for exm, pred_logit in zip(inputs_dct['batch_ner_exm'], batch_predict_lst):
                    # pred_prob = torch.sigmoid(pred)  # prob
                    exm.distilled_span_ner_pred_lst = pred_logit.numpy()  # [*,ent]  # ent learned so far the previous task 截至到当前任务的实体
                iterator.set_description(f'Task{task_id} Distilling Train set Step{i} | Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1: {f1_meaner.f1:.3f}')

            if args.distill_dev:  # try to also distill Dev set to imporve the measure accuracy, not boost much, not use in published paper.
                iterator = tqdm(dev_dataloader, dynamic_ncols=True)
                curr_task_ent = [id2ent[eid] for eid in loader.tid2entids[task_id]]
                for i, inputs_dct in enumerate(iterator):  # iter steps
                    # inputs_dct.pop('span_ner_tgt_lst')
                    inputs_dct.pop('batch_span_tgt_lst_distilled')  # no need to return kl_loss
                    seq_len = inputs_dct['ori_seq_len']
                    with torch.no_grad():  #
                        batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, last_task_id, mode='test')
                    batch_predict = batch_predict.detach().cpu()
                    f1_meaner.add(*detail_f1)
                    batch_predict_lst = torch.split(batch_predict, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]
                    for exm, length, pred in zip(inputs_dct['batch_ner_exm'], seq_len.tolist(), batch_predict_lst):
                        exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(torch.sigmoid(pred).numpy(), length, id2ent, threshold=0.5)  # sigmoid
                        flat_pred_ent_dct = exm.get_flat_pred_ent_dct()
                        delattr(exm, 'pred_ent_dct')
                        if not hasattr(exm, 'ori_ent_dct'):
                            exm.ori_ent_dct = copy.deepcopy(exm.ent_dct)
                        exm.ent_dct = {ent: v for ent, v in exm.ori_ent_dct.items() if ent in curr_task_ent}  # curr task ents
                        exm.ent_dct.update(flat_pred_ent_dct)  # last task so far ents

                    for exm, pred_logit in zip(inputs_dct['batch_ner_exm'], batch_predict_lst):   # 先前任务的蒸馏结果保存（对于任务0，没有；对于任务1，保存ORG伪标签；对于任务2，保存ORG、PERSON伪标签；以此类推）
                        # pred_prob = torch.sigmoid(pred)  # prob
                        exm.distilled_span_ner_pred_lst = pred_logit.numpy()  # [*,ent]  # ent learned so far the previous task 截至到当前任务的实体
                    iterator.set_description(f'Task{task_id} Distilling Dev   set| Step{i} | Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1: {f1_meaner.f1:.3f}')

        if args.use_best_dev:
            best_dev_f1 = -1.
        model.init_opt()
        model.init_lrs(num_step_per_epo=len(train_dataloader), epo=args.num_epochs, num_warmup_steps=args.warmup_step)

        step_in_task = 0
        for ep in range(args.num_epochs):
            model.ep = ep
            f1_meaner = utils.F1_Meaner()
            model.train()
            logger.info(utils.header_format(f'task {task_id} train epo {ep}', sep='='))
            iterator = tqdm(train_dataloader, ncols=300, dynamic_ncols=True)    # 在进行数据集处理时，我们完全没有必要一开始就把所有的数据全部处理好，特别是非常大的数据集。完全可以采用一种“lazy dataset”的方式，在训练用到一批时我再加载一批。
            for i, inputs_dct in enumerate(iterator):  # iter steps  # 从这里开始才进行对token变为input_ids的操作，具体为train_dataloader里的dataset是LazyDataset的实例化，LazyDataset中的__getitem__方法中对token变为input_ids
                step_in_task += 1
                if learn_mode == 'cl':
                    loss, span_loss, sparse_loss, kl_loss = model.observe(inputs_dct, task_id, f1_meaner)
                if learn_mode == 'non_cl':
                    loss, span_loss, sparse_loss, kl_loss = model.observe_non_cl(inputs_dct, task_id, f1_meaner)
                opengate = ' '.join([f'{g}({g / (model.task_embed.shape[-1]):.3f})' for g in (model.task_embed.detach() > 0.).sum(-1).tolist()]) if args.use_task_embed else ''
                iterator.set_description(
                    f'Task{task_id} Train Ep {ep}/{args.num_epochs} Step{i} | '
                    f'Loss:{loss:.3f} ({span_loss:.3f} {sparse_loss:.3f} {kl_loss:.3f} {float(model.entropy_loss):.3f}) | '
                    f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} | '
                    f'BS:{train_dataloader.batch_size} LR:{model.curr_lr:.6f} '
                    f'gnorm:{model.total_norm:.3f} gclip:{model.grad_clip} '
                    f'OpenGate:{opengate}'
                )
            # evalate
            utils.save_args_to_json_file(args, f'{args.curr_ckpt_dir}/args.json')
            if args.use_best_dev:
                if learn_mode == 'cl':
                    if args.distill_dev:
                        m = spankl_eval(model, dev_dataloader, task_id, loader, 'Dev  ', mode='so_far')
                    else:
                        m = spankl_eval(model, dev_dataloader, task_id, loader, 'Dev  ', mode='curr')
                if learn_mode == 'non_cl':
                    m = spankl_eval(model, dev_dataloader, task_id, loader, 'Dev  ', mode='so_far')  # non_cl
                f1 = m['maf1']
                metrics['dev_metric'][task_id][ep] = f1
                if f1 > best_dev_f1:
                    best_dev_f1 = f1
                    metrics["task_best_dev_epo"][task_id] = ep
                    model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save best dev task_{task_id}_model.pt in epo: {ep}')
            else:
                model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save model.pt in task: {task_id} epo: {ep}')

            if args.test_per_epo:
                m = spankl_eval(model, test_dataloader_All, task_id, loader, info_str='Test All', mode='so_far')
                metrics['test_metric'][task_id][ep] = m['detail_stat']
                metrics['test_metric'][task_id][ep]['micro_f1'] = m['mif1']
                metrics['test_metric'][task_id][ep]['macro_f1'] = m['maf1']
                if args.also_Test_Filter:
                    m = spankl_eval(model, test_tasks_dataloaders_Filter[task_id], task_id, loader, info_str='Test Filter', mode='so_far')
                    metrics['filter_test_metric'][task_id][ep] = m['detail_stat']
                    metrics['filter_test_metric'][task_id][ep]['micro_f1'] = m['mif1']
                    metrics['filter_test_metric'][task_id][ep]['macro_f1'] = m['maf1']
                utils.save_json(metrics, f'{args.curr_ckpt_dir}/overview_metric.json')
                simply_print_cl_metric(args.curr_ckpt_dir)

        if not args.test_per_epo:  # test performance finally
            if args.use_best_dev:  # load best dev to test
                model.load_model(f'{args.curr_ckpt_dir}/task_{task_id}_model.pt', info=f'loaded best dev model to test! task_best_dev_epo: {metrics["task_best_dev_epo"][task_id]}')  # obtain best model
            m = spankl_eval(model, test_dataloader_All, task_id, loader, info_str='Test All', mode='so_far',
                            saving_exm_file=f'{args.curr_ckpt_dir}/testAll_exm_lst{task_id}.jsonl')
            metrics['test_metric'][task_id][ep] = m['detail_stat']
            metrics['test_metric'][task_id][ep]['micro_f1'] = m['mif1']
            metrics['test_metric'][task_id][ep]['macro_f1'] = m['maf1']
            if args.also_Test_Filter:
                m = spankl_eval(model, test_tasks_dataloaders_Filter[task_id], task_id, loader, info_str='Test Filter', mode='so_far',
                                saving_exm_file=f'{args.curr_ckpt_dir}/testFilter_exm_lst{task_id}.jsonl')
                metrics['filter_test_metric'][task_id][ep] = m['detail_stat']
                metrics['filter_test_metric'][task_id][ep]['micro_f1'] = m['mif1']
                metrics['filter_test_metric'][task_id][ep]['macro_f1'] = m['maf1']
        utils.save_json(metrics, f'{args.curr_ckpt_dir}/overview_metric.json')
        simply_print_cl_metric(args.curr_ckpt_dir)

        # save task_embed vector as .npz 保存task_embed向量到npz中
        if args.use_task_embed:
            np.savez(args.curr_ckpt_dir / '/task_embed.npz', model.task_embed.detach().cpu().numpy())

    return None


def spankl_ner_all_tasks(model: modules.SpanKL, loader: ner_loader, args):
    """
    A experimental setting to verify the Standard Supervised Full Data performance.
    Not equal to CL or Non_C, because it only learn once by full data
    """
    train_dataloader = loader.get_task_dataloader(mode='train')  # all task
    dev_dataloader = loader.get_task_dataloader(mode='dev')  # all task
    test_dataloader = loader.get_task_dataloader(mode='test')
    args.test_per_epo = True

    task_best_dev_epo = []

    metric = {'test_metric': {},
              'dev_metric': {},
              'train_metric': {},
              }
    task_id = loader.num_tasks - 1  # pretend to be the last task
    print('=' * 40)
    metric['test_metric'][task_id] = {}
    metric['dev_metric'][task_id] = {}
    metric['train_metric'][task_id] = {}

    if args.use_best_dev:
        best_dev_f1 = -1.
        best_dev_epo = -1
    model.init_opt()
    model.init_lrs(num_step_per_epo=len(train_dataloader), epo=args.num_epochs, num_warmup_steps=args.warmup_step)

    for ep in range(args.num_epochs):
        f1_meaner = utils.F1_Meaner()
        model.train()
        print('=' * 20)
        prog_bar = tqdm(train_dataloader, ncols=300, dynamic_ncols=True)
        for i, inputs_dct in enumerate(prog_bar):  # iter step
            loss, span_loss, sparse_loss, kl_loss = model.observe_all(inputs_dct, f1_meaner)
            prog_bar.set_description(
                f'Task:{task_id} | Train | Epoch: {ep}/{args.num_epochs} | Iter:{i} | '
                f'Loss:{loss:.3f} ({span_loss:.3f} {sparse_loss:.3f} {kl_loss:.3f}) | '
                f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} | '
                f'LR:{model.curr_lr:.6f} gnorm:{model.total_norm:.3f} |'
                f'BS:{train_dataloader.batch_size} gclip:{model.grad_clip}'
            )
        utils.save_args_to_json_file(args, f'{args.curr_ckpt_dir}/args.json')
        if args.use_best_dev:
            f1, _, raw_f1 = spankl_eval(model, dev_dataloader, task_id, loader, mode='test')
            print(f'**Dev  | RawF1:{raw_f1} | F1:{f1}')
            metric['dev_metric'][task_id][ep] = f1
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_dev_epo = ep
                model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt')
                print(f'save_best_dev task_{task_id}_model.pt in epo: {best_dev_epo} ')
        else:
            model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt')
            print(f'save_model.pt in task: {task_id} epo: {ep} ')

        if args.test_per_epo:
            f1, detail_stat, raw_f1 = spankl_eval(model, test_dataloader, task_id, loader, mode='test')
            print(f'**Test | RawF1:{raw_f1} | F1:{f1}')
            metric['test_metric'][task_id][ep] = detail_stat
            metric['test_metric'][task_id][ep]['micro_f1'] = f1
            metric['task_best_dev_epo'] = task_best_dev_epo
            utils.save_json(metric, args.curr_ckpt_dir / f'/overview_metric.json')

    if not args.test_per_epo:
        if args.use_best_dev:  # load best dev to test
            task_best_dev_epo.append(best_dev_epo)
            print('task_best_dev_epo', task_best_dev_epo)
            model.load_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt')  # obtain best model
            print('load best dev model to test!')
        f1, detail_stat, raw_f1 = spankl_eval(model, test_dataloader, task_id, loader, mode='test')
        print(f'**Test | RawF1:{raw_f1} | F1:{f1}')
        metric['test_metric'][task_id][ep] = detail_stat
        metric['test_metric'][task_id][ep]['micro_f1'] = f1
        metric['task_best_dev_epo'] = task_best_dev_epo
        utils.save_json(metric, args.curr_ckpt_dir / f'/overview_metric.json')

    # save task_embed vector as .npz 保存task_embed向量到npz中
    if args.use_task_embed:
        np.savez(args.curr_ckpt_dir / '/task_embed.npz', model.task_embed.detach().cpu().numpy())
        # task_embed_lst.append(model.task_embed.detach().cpu().numpy())
        # np.savez(args.log_dir + f'task_embed.npz', *task_embed_lst)

    return None


def mlm_ner(model: modules.MLM, loader: ner_loader, args, learn_mode='cl'):
    # rz+
    # 加载label_list
    if args.label_map_path is not None:
        print("Loading label map from {}...".format(args.label_map_path))
        import json  # TODO
        ori_label_token_map = json.load(open(args.label_map_path, 'r'))
    else:
        ori_label_token_map = {"I-PER":['Michael', 'John', 'David', 'Mark', 'Martin', 'Paul'], "I-ORG": ['Inc', 'Co', 'Corp', 'Party', 'Association', '&'],
                           "I-LOC":['Germany', 'Australia', 'England', 'France', 'Italy', 'Belgium'], "I-MISC":['German', 'Cup', 'French', 'Israeli', 'Australian', 'Olympic']}

    label_list = list(ori_label_token_map.keys())
    label_list += 'O'

    # label_to_id = {"O":0}
    # for l in label_list:
    #     if l != "O":
    #         label_to_id[l] = len(label_to_id)
    # # label_to_id = {l: i for i, l in enumerate(label_list)}
    # num_labels = len(label_list)
    # print(ori_label_token_map)
    # if True:  # args.eval_label_schema == "BIO":
    #     import copy
    #     new_label_to_id = copy.deepcopy(label_to_id)
    #     for label, id in label_to_id.items():
    #         if label != "O" and "B-"+label[2:] not in label_to_id:
    #             new_label_to_id["B-"+label[2:]] = len(new_label_to_id)
    #     label_to_id = new_label_to_id
    # id_to_label = {id:label for label,id in label_to_id.items()}
    # print(label_to_id)

    if "roberta" in args.bert_model_dir:
        tokenizer = add_label_token_roberta(model, loader, ori_label_token_map)
    elif "bert" in args.bert_model_dir:
        tokenizer = add_label_token_bert(model, loader, ori_label_token_map)
    else:
        pass

    label_token_map = {item:item for item in ori_label_token_map}
    # label_token_map = ori_label_token_map
    print(ori_label_token_map)
    label_token_to_id = {label: tokenizer.convert_tokens_to_ids(label_token) for label, label_token in label_token_map.items()}
    print(label_token_to_id)
    label_token_id_to_label = {idx:label for label,idx in label_token_to_id.items()}
    # rz+

    test_dataloader_All: DataLoader = loader.get_task_dataloader(mode='test')
    test_tasks_dataloaders_Filter: List[DataLoader] = loader.test_tasks_dataloaders_filtered
    id2ent = loader.datareader.id2ent
    # tb_writer = SummaryWriter(f'./runs/{args.corpus}-{args.time_series}')
    args.test_per_epo = False
    args.use_finetune = False

    start_taskid = 0
    start_model = ''
    end_taskid = 0
    # start_taskid = 5
    # start_model = 'onto-0-2022-07-21_10-02-57-1824-spankl_split_perm0'
    # end_taskid = 1

    metrics = {'test_metric': {},
               'dev_metric': {},
               'train_metric': {},
               'task_best_dev_epo': [-1] * loader.num_tasks,
               'filter_test_metric': {}
               }
    if start_model and os.path.exists(args.ckpt_dir / f'{start_model}/overview_metric.json'):
        metrics = utils.load_json(args.ckpt_dir / f'{start_model}/overview_metric.json')
    model.to(args.device)  # TODO 这里加了前面就不要了
    # TODO 写优雅
    # def freeze_model(model):
    #     # 冻结除了 embedding 层、layernorm 层和 cls 层外的所有层
    #     for param in model.parameters():
    #         param.requires_grad = False
    #
    #     # 解冻 embedding 层、layernorm 层和 cls 层
    #     for name, param in model.named_parameters():
    #         # if 'embeddings' in name or 'LayerNorm' in name or 'cls' in name:
    #         if 'cls' in name:
    #             param.requires_grad = True

    # freeze_model(model.bert_layer)
    for task_id in range(loader.num_tasks):
        if start_model and start_taskid and task_id < start_taskid: continue
        if end_taskid and task_id >= end_taskid: break
        logger.info(utils.header_format(f'task {task_id} train', sep='='))
        metrics['test_metric'][task_id] = {}
        metrics['dev_metric'][task_id] = {}
        metrics['train_metric'][task_id] = {}
        metrics['filter_test_metric'][task_id] = {}

        if learn_mode == 'cl':
            train_dataloader = loader.get_task_dataloader(mode='train', tid=task_id)   # 只包含task_id=0任务的train_dataloader
            dev_dataloader = loader.get_task_dataloader(mode='dev', tid=task_id)
        if learn_mode == 'non_cl':
            train_dataloader = loader.so_far_train_tasks_dataloaders[task_id]
            dev_dataloader = loader.so_far_dev_tasks_dataloaders[task_id]

        # load possible last task model (the last model or the best model (use dev))
        if task_id > 0:
            if start_model and start_taskid and task_id == start_taskid:
                model.load_model(args.ckpt_dir / f'{start_model}/task_{start_taskid - 1}_model.pt',
                                 info=f'load existing start model:{start_model} task:{start_taskid - 1} success!')
            else:
                model.load_model(args.curr_ckpt_dir / f'task_{task_id - 1}_model.pt',
                                 info=f'load prev task model: task_{task_id - 1}_model.pt. last best_dev_epo: {metrics["task_best_dev_epo"][task_id - 1]}')

        # distill the knowledge from the previous model
        if learn_mode == 'cl' and args.use_distill and task_id > 0:
            # f1_meaner = utils.F1_Meaner()
            last_task_id = task_id - 1
            model.eval()
            # distill Train set
            iterator = tqdm(train_dataloader, dynamic_ncols=True)
            [delattr(exm, 'distilled_task_ent_output') for exm in train_dataloader.dataset.instances if hasattr(exm, 'distilled_task_ent_output')]
            for i, inputs_dct in enumerate(iterator):  # iter steps
                # inputs_dct.pop('span_ner_tgt_lst')
                # inputs_dct.pop('batch_span_tgt_lst_distilled')  # no need to return kl_loss
                inputs_dct.pop('batch_distilled_task_ent_output')  # no need to return kl_loss
                seq_len = inputs_dct['ori_seq_len']
                with torch.no_grad():
                    # batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, last_task_id, mode='test')
                    preds, refs, tokens, entlm_outputs_prob, batch_entlm_logits_list, f1, detail_f1 = model(inputs_dct, last_task_id, mode='test')
                # batch_predict = preds.detach().cpu()
                # f1_meaner.add(*detail_f1)
                # batch_predict_lst = torch.split(batch_predict, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]

                # 下面是尝试使用logits蒸馏方法，但是mlm这种范式不可以使用logits蒸馏，因为没有输出头
                batch_entlm_logits_list = [logits.half() for logits in batch_entlm_logits_list]
                for i, (exm, pred_logit) in enumerate(zip(inputs_dct['batch_ner_exm'], batch_entlm_logits_list)):
                    # pred_prob = torch.sigmoid(pred)  # prob
                    # exm.distilled_task_ent_output = pred_logit.detach().cpu().numpy()  # 这种转为numpy后会因为内存占用太大而报错Process finished with exit code 137

                    # 只蒸馏当前任务logits TODO: 灵活蒸馏
                    # selected_dimensions = torch.cat([pred_logit[:, :28996], pred_logit[:, 29000:29001]], dim=1)   # list(model.tokenizer.vocab.keys())[list(model.tokenizer.vocab.values()).index(28996)]
                    last_task_ofe = len(model.tokenizer.vocab) - sum(loader.num_ents_per_task) + sum(loader.num_ents_per_task[:task_id])
                    last_task_logits = pred_logit[:, :last_task_ofe]   # last_task_ofe=28997代表取包含ORG（ids为28996）任务的头
                    selected_dimensions = last_task_logits.view(pred_logit.size(0), -1)

                    exm.distilled_task_ent_output = selected_dimensions.detach().cpu().numpy()  # [*,ent]  # ent learned so far the previous task 截至到当前任务的实体
                # iterator.set_description(f'Task{task_id} Distilling Train set Step{i} | Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1: {f1_meaner.f1:.3f}')
                iterator.set_description(f'Task{task_id} Distilling Train set Step{i}')

                # for exm, self_pred in zip(inputs_dct['batch_ner_exm'], preds):
                #     # pred_prob = torch.sigmoid(pred)  # prob
                #     # exm.distilled_task_ent_output = pred_logit.detach().cpu().numpy()  # 这种转为numpy后会因为内存占用太大而报错Process finished with exit code 137
                #     exm.distilled_task_ent_output = self_pred  # [*,ent]  # ent learned so far the previous task 截至到当前任务的实体
                # # iterator.set_description(f'Task{task_id} Distilling Train set Step{i} | Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1: {f1_meaner.f1:.3f}')
                # iterator.set_description(f'Task{task_id} Distilling Train set Step{i}')

            if args.distill_dev:  # try to also distill Dev set to imporve the measure accuracy, not boost much, not use in published paper.
                iterator = tqdm(dev_dataloader, dynamic_ncols=True)
                curr_task_ent = [id2ent[eid] for eid in loader.tid2entids[task_id]]
                for i, inputs_dct in enumerate(iterator):  # iter steps
                    # inputs_dct.pop('span_ner_tgt_lst')
                    inputs_dct.pop('batch_span_tgt_lst_distilled')  # no need to return kl_loss
                    seq_len = inputs_dct['ori_seq_len']
                    with torch.no_grad():  #
                        batch_predict, f1, detail_f1, span_loss, kl_loss = model(inputs_dct, last_task_id, mode='test')
                    batch_predict = batch_predict.detach().cpu()
                    f1_meaner.add(*detail_f1)
                    batch_predict_lst = torch.split(batch_predict, (seq_len * (seq_len + 1) / 2).int().tolist())  # 根据每个batch中的样本拆开 list of [num_spans, ent]
                    for exm, length, pred in zip(inputs_dct['batch_ner_exm'], seq_len.tolist(), batch_predict_lst):
                        exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(torch.sigmoid(pred).numpy(), length, id2ent, threshold=0.5)  # sigmoid
                        flat_pred_ent_dct = exm.get_flat_pred_ent_dct()
                        delattr(exm, 'pred_ent_dct')
                        if not hasattr(exm, 'ori_ent_dct'):
                            exm.ori_ent_dct = copy.deepcopy(exm.ent_dct)
                        exm.ent_dct = {ent: v for ent, v in exm.ori_ent_dct.items() if ent in curr_task_ent}  # curr task ents
                        exm.ent_dct.update(flat_pred_ent_dct)  # last task so far ents

                    for exm, pred_logit in zip(inputs_dct['batch_ner_exm'], batch_predict_lst):   # 先前任务的蒸馏结果保存（对于任务0，没有；对于任务1，保存ORG伪标签；对于任务2，保存ORG、PERSON伪标签；以此类推）
                        # pred_prob = torch.sigmoid(pred)  # prob
                        exm.distilled_span_ner_pred_lst = pred_logit.numpy()  # [*,ent]  # ent learned so far the previous task 截至到当前任务的实体
                    iterator.set_description(f'Task{task_id} Distilling Dev   set| Step{i} | Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1: {f1_meaner.f1:.3f}')

        if args.use_best_dev:
            # best_dev_f1 = -1.
            best_dev_f1 = 0.  # rz改  TODO 源代码为-1，为什么我的必须是0

        def freeze_model(model, freeze_layers=None):
            if freeze_layers:
                for param in model.bert.encoder.layer[:freeze_layers].parameters():
                    param.requires_grad = False
        # rz+
        if task_id == 0:
            current_num_epochs = args.t0_num_epochs  # Set epochs to 1 for the first task
            # freeze_model(model.bert_layer, freeze_layers=3)
        else:
            current_num_epochs = args.num_epochs
            freeze_model(model.bert_layer, freeze_layers=9)   # TODO 到时候把冻结改为和下面一样的.init_opt()一样的，写在NERmodel类方法里
        # rz+

        model.init_opt()
        # model.init_lrs(num_step_per_epo=len(train_dataloader), epo=args.num_epochs, num_warmup_steps=args.warmup_step)
        model.init_lrs(num_step_per_epo=len(train_dataloader), epo=current_num_epochs, num_warmup_steps=args.warmup_step)

        step_in_task = 0
        # for ep in range(args.num_epochs):
        for ep in range(current_num_epochs):
            model.ep = ep
            f1_meaner = utils.F1_Meaner()
            model.train()
            logger.info(utils.header_format(f'task {task_id} train epo {ep}', sep='='))
            iterator = tqdm(train_dataloader, ncols=300, dynamic_ncols=True)    # 在进行数据集处理时，我们完全没有必要一开始就把所有的数据全部处理好，特别是非常大的数据集。完全可以采用一种“lazy dataset”的方式，在训练用到一批时我再加载一批。
            for i, inputs_dct in enumerate(iterator):  # iter steps  # 从这里开始才进行对token变为input_ids的操作，具体为train_dataloader里的dataset是LazyDataset的实例化，LazyDataset中的__getitem__方法中对token变为input_ids
                step_in_task += 1
                if learn_mode == 'cl':
                    # loss, span_loss, sparse_loss, kl_loss = model.observe(inputs_dct, task_id, f1_meaner)
                    loss, ce_loss, mse_loss, kl_loss = model.observe_mlm(inputs_dct, task_id, f1_meaner)
                if learn_mode == 'non_cl':
                    loss, span_loss, sparse_loss, kl_loss = model.observe_non_cl(inputs_dct, task_id, f1_meaner)
                opengate = ' '.join([f'{g}({g / (model.task_embed.shape[-1]):.3f})' for g in (model.task_embed.detach() > 0.).sum(-1).tolist()]) if args.use_task_embed else ''
                iterator.set_description(
                    # f'Task{task_id} Train Ep {ep}/{args.num_epochs} Step{i} | '
                    f'Task{task_id} Train Ep {ep}/{current_num_epochs} Step{i} | '
                    # f'Loss:{loss:.3f} ({span_loss:.3f} {sparse_loss:.3f} {kl_loss:.3f} {float(model.entropy_loss):.3f}) | '
                    f'Loss:{loss:.3f} ({ce_loss:.3f} {mse_loss:.3f} {kl_loss:.3f} {float(model.entropy_loss):.3f}) | '
                    f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} | '
                    f'BS:{train_dataloader.batch_size} LR:{model.curr_lr:.6f} '
                    f'gnorm:{model.total_norm:.3f} gclip:{model.grad_clip} '
                    f'OpenGate:{opengate}'
                )
            # evalate
            utils.save_args_to_json_file(args, f'{args.curr_ckpt_dir}/args.json')
            if args.use_best_dev:
                if learn_mode == 'cl':
                    if args.distill_dev:
                        m = mlm_eval(model, dev_dataloader, task_id, loader, 'Dev  ', mode='so_far')
                    else:
                        m = mlm_eval(model, dev_dataloader, task_id, loader, 'Dev  ', mode='curr')
                if learn_mode == 'non_cl':
                    m = mlm_eval(model, dev_dataloader, task_id, loader, 'Dev  ', mode='so_far')  # non_cl
                f1 = m['maf1']
                metrics['dev_metric'][task_id][ep] = f1
                if f1 > best_dev_f1:
                    best_dev_f1 = f1
                    metrics["task_best_dev_epo"][task_id] = ep
                    model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save_best_dev task_{task_id}_model.pt in epo: {ep} ')

            else:
                model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save_model.pt in task: {task_id} epo: {ep} ')

            #     f1 = m['maf1']
            #     metrics['dev_metric'][task_id][ep] = f1
            #     if f1 > best_dev_f1:
            #         best_dev_f1 = f1
            #         metrics["task_best_dev_epo"][task_id] = ep
            #         model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save best dev task_{task_id}_model.pt in epo: {ep}')
            # else:
            #     model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt', info=f'save model.pt in task: {task_id} epo: {ep}')
            #
            # if args.test_per_epo:
            #     m = mlm_eval(model, test_dataloader_All, task_id, loader, info_str='Test All', mode='so_far')
            #     metrics['test_metric'][task_id][ep] = m['detail_stat']
            #     metrics['test_metric'][task_id][ep]['micro_f1'] = m['mif1']
            #     metrics['test_metric'][task_id][ep]['macro_f1'] = m['maf1']
            #     if args.also_Test_Filter:
            #         m = mlm_eval(model, test_tasks_dataloaders_Filter[task_id], task_id, loader, info_str='Test Filter', mode='so_far')
            #         metrics['filter_test_metric'][task_id][ep] = m['detail_stat']
            #         metrics['filter_test_metric'][task_id][ep]['micro_f1'] = m['mif1']
            #         metrics['filter_test_metric'][task_id][ep]['macro_f1'] = m['maf1']
            #     utils.save_json(metrics, f'{args.curr_ckpt_dir}/overview_metric.json')
            #     simply_print_cl_metric(args.curr_ckpt_dir)

        if not args.test_per_epo:  # test performance finally
            if args.use_best_dev:  # load best dev to test
                model.load_model(f'{args.curr_ckpt_dir}/task_{task_id}_model.pt', info=f'loaded best dev model to test! task_best_dev_epo: {metrics["task_best_dev_epo"][task_id]}')  # obtain best model
            m = mlm_eval(model, test_dataloader_All, task_id, loader, info_str='Test All', mode='so_far',
                            saving_exm_file=f'{args.curr_ckpt_dir}/testAll_exm_lst{task_id}.jsonl')
            metrics['test_metric'][task_id][ep] = m['detail_stat']
            metrics['test_metric'][task_id][ep]['micro_f1'] = m['mif1']
            metrics['test_metric'][task_id][ep]['macro_f1'] = m['maf1']
            if args.also_Test_Filter:
                m = mlm_eval(model, test_tasks_dataloaders_Filter[task_id], task_id, loader, info_str='Test Filter', mode='so_far',
                                saving_exm_file=f'{args.curr_ckpt_dir}/testFilter_exm_lst{task_id}.jsonl')
                metrics['filter_test_metric'][task_id][ep] = m['detail_stat']
                metrics['filter_test_metric'][task_id][ep]['micro_f1'] = m['mif1']
                metrics['filter_test_metric'][task_id][ep]['macro_f1'] = m['maf1']
        utils.save_json(metrics, f'{args.curr_ckpt_dir}/overview_metric.json')
        simply_print_cl_metric(args.curr_ckpt_dir)

        # save task_embed vector as .npz 保存task_embed向量到npz中
        if args.use_task_embed:
            np.savez(args.curr_ckpt_dir / '/task_embed.npz', model.task_embed.detach().cpu().numpy())

    return None


def mlm_ner_all_tasks(model: modules.MLM, loader: ner_loader, args):
    """
    A experimental setting to verify the Standard Supervised Full Data performance.
    Not equal to CL or Non_C, because it only learn once by full data
    """
    train_dataloader = loader.get_task_dataloader(mode='train')  # all task
    dev_dataloader = loader.get_task_dataloader(mode='dev')  # all task
    test_dataloader = loader.get_task_dataloader(mode='test')
    args.test_per_epo = True

    task_best_dev_epo = []

    metric = {'test_metric': {},
              'dev_metric': {},
              'train_metric': {},
              }
    task_id = loader.num_tasks - 1  # pretend to be the last task
    print('=' * 40)
    metric['test_metric'][task_id] = {}
    metric['dev_metric'][task_id] = {}
    metric['train_metric'][task_id] = {}

    if args.use_best_dev:
        best_dev_f1 = -1.
        best_dev_epo = -1
    model.init_opt()
    model.init_lrs(num_step_per_epo=len(train_dataloader), epo=args.num_epochs, num_warmup_steps=args.warmup_step)

    for ep in range(args.num_epochs):
        f1_meaner = utils.F1_Meaner()
        model.train()
        print('=' * 20)
        prog_bar = tqdm(train_dataloader, ncols=300, dynamic_ncols=True)
        for i, inputs_dct in enumerate(prog_bar):  # iter step
            loss, span_loss, sparse_loss, kl_loss = model.observe_all(inputs_dct, f1_meaner)
            prog_bar.set_description(
                f'Task:{task_id} | Train | Epoch: {ep}/{args.num_epochs} | Iter:{i} | '
                f'Loss:{loss:.3f} ({span_loss:.3f} {sparse_loss:.3f} {kl_loss:.3f}) | '
                f'Prec:{f1_meaner.prec:.3f} Rec:{f1_meaner.rec:.3f} F1:{f1_meaner.f1:.3f} | '
                f'LR:{model.curr_lr:.6f} gnorm:{model.total_norm:.3f} |'
                f'BS:{train_dataloader.batch_size} gclip:{model.grad_clip}'
            )
        utils.save_args_to_json_file(args, f'{args.curr_ckpt_dir}/args.json')
        if args.use_best_dev:
            f1, _, raw_f1 = mlm_eval(model, dev_dataloader, task_id, loader, mode='test')
            print(f'**Dev  | RawF1:{raw_f1} | F1:{f1}')
            metric['dev_metric'][task_id][ep] = f1
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_dev_epo = ep
                model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt')
                print(f'save_best_dev task_{task_id}_model.pt in epo: {best_dev_epo} ')
        else:
            model.save_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt')
            print(f'save_model.pt in task: {task_id} epo: {ep} ')

        if args.test_per_epo:
            f1, detail_stat, raw_f1 = mlm_eval(model, test_dataloader, task_id, loader, mode='test')
            print(f'**Test | RawF1:{raw_f1} | F1:{f1}')
            metric['test_metric'][task_id][ep] = detail_stat
            metric['test_metric'][task_id][ep]['micro_f1'] = f1
            metric['task_best_dev_epo'] = task_best_dev_epo
            utils.save_json(metric, args.curr_ckpt_dir / f'/overview_metric.json')

    if not args.test_per_epo:
        if args.use_best_dev:  # load best dev to test
            task_best_dev_epo.append(best_dev_epo)
            print('task_best_dev_epo', task_best_dev_epo)
            model.load_model(args.curr_ckpt_dir / f'task_{task_id}_model.pt')  # obtain best model
            print('load best dev model to test!')
        f1, detail_stat, raw_f1 = mlm_eval(model, test_dataloader, task_id, loader, mode='test')
        print(f'**Test | RawF1:{raw_f1} | F1:{f1}')
        metric['test_metric'][task_id][ep] = detail_stat
        metric['test_metric'][task_id][ep]['micro_f1'] = f1
        metric['task_best_dev_epo'] = task_best_dev_epo
        utils.save_json(metric, args.curr_ckpt_dir / f'/overview_metric.json')

    # save task_embed vector as .npz 保存task_embed向量到npz中
    if args.use_task_embed:
        np.savez(args.curr_ckpt_dir / '/task_embed.npz', model.task_embed.detach().cpu().numpy())
        # task_embed_lst.append(model.task_embed.detach().cpu().numpy())
        # np.savez(args.log_dir + f'task_embed.npz', *task_embed_lst)

    return None


def train_main(args):
    args.time_series = utils.get_curr_time_str('%Y-%m-%d_%H-%M-%S-%f')[:-2]  # e.g., 2022-06-05_23-39-12-0144
    args.info = f'{args.m}_{args.setup}{"_non_cl" if args.non_cl else ""}_{args.perm}'
    args.curr_ckpt_dir = args.ckpt_dir / f'{args.corpus}-{args.seed}-{args.time_series}-{args.info}'

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # '-1' to use CPU
    if args.gpu == '-1':
        args.device = torch.device('cpu')
    else:
        if args.gpu == '-2':
            args.device = utils.auto_device(logger, torch, np, subprocess)  # may be false to any GPUs
        else:
            args.device = torch.device(f'cuda:{args.gpu}')
    args.use_gpu = False
    if args.device.type.startswith('cuda'):
        torch.cuda.set_device(args.device)
        args.use_gpu = True

    logger.info(f'truly used gpu: {args.device}')

    # utils.print_vars(args, maxlen=200)
    logger.info(" ".join(sys.argv))
    logger.info(f'args:\n%s', pprint.pformat(args.__dict__))
    utils.setup_seed(args.seed, np, torch)
    logger.info(utils.header_format(f"Training {args.m} {args.corpus}", sep='='))

    if args.corpus == 'fewnerd':
        loader = ner_loader.FewNERD_Loader(args=args)
    elif args.corpus == 'conll':
        loader = ner_loader.Conll_Loader(args=args)
    elif args.corpus == 'onto':
        loader = ner_loader.Onto_Loader(args=args)
    elif args.corpus == 'onto_fs':
        loader = ner_loader.Onto_fs_Loader(args=args)
    else:
        raise NotImplementedError
    arch = 'span' if args.m == 'spankl' else 'seq'
    loader.init_data(bsz=args.batch_size, quick_test=args.quick_test, arch=arch,
                     use_pt=args.pretrain_mode == 'feature_based', gpu=args.use_gpu)

    """ perm task """
    # if args.perm == 'perm0':
    #     loader.permute_task_order(None)  # None = not permuate
    # else:
    #     loader.permute_task_order(list(map(int, args.perm_ids)))

    # load model
    model = {
        'mlm': modules.MLM,
        'spankl': modules.SpanKL,
        'add': modules.BaselineAdd,
        'ext': modules.BaselineExtend,
    }[args.m](args, loader)

    # model.to(args.device)

    if args.m == 'mlm':
        if args.non_cl:
            mlm_ner(model, loader, args, learn_mode='non_cl')
        elif args.all_tasks:
            mlm_ner_all_tasks(model, loader, args)
        else:
            mlm_ner(model, loader, args, learn_mode='cl')
    if args.m == 'spankl':
        if args.non_cl:
            spankl_ner(model, loader, args, learn_mode='non_cl')
        elif args.all_tasks:
            spankl_ner_all_tasks(model, loader, args)
        else:
            spankl_ner(model, loader, args, learn_mode='cl')
    if args.m in ['add', 'ext']:
        if args.non_cl:
            baseline_ner(model, loader, args, args.m, learn_mode='non_cl')
        else:
            baseline_ner(model, loader, args, args.m, learn_mode='cl')

    # 只保留最后任务的模型节省空间
    # maxtasks = 6 if args.corpus == 'onto' else 8
    # for tid in range(maxtasks - 1):
    #     model_path = args.curr_ckpt_dir / f'task_{tid}_model.pt'
    #     if os.path.exists(model_path): os.remove(model_path)
    exit(0)


def eval_ckpt(args, existed_model_ckpt_dir, save_res_in_new=False):
    """ Baseline or SpanKL model path """
    model_ckpt_dir = args.ckpt_dir / Path(existed_model_ckpt_dir)
    if save_res_in_new:
        args.time_series = utils.get_curr_time_str('%Y-%m-%d_%H-%M-%S-%f')[:-2]  # e.g., 2022-06-05_23-39-12-0144
        args.info = '-' + args.info if args.info else ''  # prefix '-' info
        saved_ckpt_dir = args.ckpt_dir / f'{args.corpus}-{args.seed}-{args.time_series}{args.info}'
    else:
        saved_ckpt_dir = model_ckpt_dir

    args = utils.load_args_by_json_file(model_ckpt_dir / 'args.json', exist_args=args)  # load existed args
    if args.gpu == '-1':
        args.use_gpu = False
        args.device = torch.device('cpu')
    else:
        args.use_gpu = True
        if args.gpu == '-2':
            import subprocess

            args.device = utils.auto_device(logger, torch, np, subprocess)
        else:
            args.device = torch.device(f'cuda:{args.gpu}')
    if args.device.type.startswith('cuda'):
        torch.cuda.set_device(args.device)
    logger.info(f'truly used gpu: {args.device}')

    logger.info(utils.header_format("Starting", sep='='))
    logger.info(" ".join(sys.argv))
    logger.info(f'args:\n%s', pprint.pformat(args.__dict__))
    logger.info(utils.header_format(f"Eval ckpt {model_ckpt_dir}", sep='='))

    utils.setup_seed(args.seed, np, torch)

    model_type = args.m
    if model_type == 'ext':
        test_fn = baseline_ext_eval
    if model_type == 'add':
        test_fn = baseline_add_eval
    if model_type == 'spankl':
        test_fn = spankl_eval
    if model_type == 'mlm':
        test_fn = mlm_eval

    if args.corpus == 'fewnerd':
        loader = ner_loader.FewNERD_Loader(args=args)
    elif args.corpus == 'onto':
        loader = ner_loader.Onto_Loader(args=args)
    elif args.corpus == 'onto_fs':
        loader = ner_loader.Onto_fs_Loader(args=args)
    else:
        raise NotImplementedError
    arch = 'span' if model_type == 'spankl' else 'seq'

    loader.init_data(bsz=args.batch_size, quick_test=args.quick_test, arch=arch,
                     use_pt=args.pretrain_mode == 'feature_based', gpu=args.use_gpu)

    # load model
    model = {
        'mlm': modules.MLM,
        'spankl': modules.SpanKL,
        'add': modules.BaselineAdd,
        'ext': modules.BaselineExtend,
    }[model_type](args, loader)
    model.to(args.device)

    torch.set_printoptions(edgeitems=10)
    metrics = {'test_metric': {},
               'dev_metric': {},
               'train_metric': {},
               'task_best_dev_epo': [-1] * loader.num_tasks,
               'filter_test_metric': {}
               }

    def eval(load_tid, predict_tid, metrics):
        model.load_model(model_ckpt_dir / f'task_{load_tid}_model.pt')

        # m = test_fn(model, loader.train_tasks_dataloaders[predict_tid], predict_tid, loader, 'Train', mode='curr')
        # m = test_fn(model, loader.dev_tasks_dataloaders[predict_tid], predict_tid, loader, 'Dev', mode='curr')
        # m = test_fn(model, loader.dev_tasks_dataloaders[predict_tid], predict_tid, loader, 'Dev', mode='so_far')

        m = test_fn(model, loader.test_dataloader, predict_tid, loader, 'Test All', mode='so_far',
                    saving_exm_file=saved_ckpt_dir / f'test_exm_t{predict_tid}.jsonl',
                    save_prob_dct=saved_ckpt_dir / f'prob_dct_t{predict_tid}.pt'
                    )

        dummpy_ep = 9
        dummpy_ep = args.num_epochs - 1
        metrics['test_metric'][predict_tid] = {}
        metrics['test_metric'][predict_tid][dummpy_ep] = m['detail_stat']
        metrics['test_metric'][predict_tid][dummpy_ep]['micro_f1'] = m['mif1']
        metrics['test_metric'][predict_tid][dummpy_ep]['macro_f1'] = m['maf1']

        m = test_fn(model, loader.test_tasks_dataloaders_filtered[predict_tid], predict_tid, loader, 'Test Filter', mode='so_far',
                    saving_exm_file=saved_ckpt_dir / f'test_filter_exm_t{predict_tid}.jsonl',
                    save_prob_dct=saved_ckpt_dir / f'prob_filter_dct_t{predict_tid}.pt'
                    )

        metrics['filter_test_metric'][predict_tid] = {}
        metrics['filter_test_metric'][predict_tid][dummpy_ep] = m['detail_stat']
        metrics['filter_test_metric'][predict_tid][dummpy_ep]['micro_f1'] = m['mif1']
        metrics['filter_test_metric'][predict_tid][dummpy_ep]['macro_f1'] = m['maf1']

    eval(5, 5, metrics)
    exit(0)

    # for tid in range(0, loader.num_tasks):
    for tid in range(0, 3):
        load_tid = tid
        predict_tid = tid
        eval(load_tid, predict_tid, metrics)

    # utils.save_json(metrics, model_ckpt_dir / 'overview_metric.json')

    # # old inspect code for task_emb
    # prev_task_embed_binary = None
    # task_num = 8 if args.corpus == 'fewnerd' else 6
    # for i in range(task_num):
    #     print('======\ntask', i)
    #     model.load_model(model_ckpt_dir / f'task_{load_tid}_model.pt')
    #     task_embed_binary = (model.task_embed > 0).float()
    #     sparse_loss = torch.norm(task_embed_binary, p=1, dim=-1) / task_embed_binary.shape[-1]
    #     print(model.task_embed.shape)
    #     print(task_embed_binary)
    #     print(model.task_embed)
    #     print(task_embed_binary.sum(dim=-1))
    #     print(sparse_loss)
    #     if prev_task_embed_binary is not None:
    #         print((task_embed_binary == prev_task_embed_binary).all(dim=-1))
    #     prev_task_embed_binary = task_embed_binary
    # # np.savez(f'{saved_ckpt_dir}/task_embed.npz', model.net.task_embed.detach().cpu().numpy())
    # ipdb.set_trace()


onto_sorted_ids_dct = {
    'perm0': [0, 1, 2, 3, 4, 5],  # normal ORG → PER → GPE → DATE → CARD → NORP
    'perm1': [3, 5, 1, 4, 0, 2],  # perm   DATE → NORP → PER → CARD → ORG → GPE # padding
    'perm2': [2, 4, 0, 5, 3, 1],  # perm   GPE → CARD → ORG → NORP → DATE → PER # padding
    'perm3': [5, 0, 3, 1, 2, 4],  # perm   NORP → ORG → DATE → PER → GPE → CARD # 平均最像
    'perm4': [4, 2, 5, 0, 1, 3],  # perm   CARD → GPE → NORP → ORG → PER → DATE # padding
    'perm5': [1, 3, 4, 2, 5, 0],  # perm   PER → DATE → CARD → GPE → NORP → ORG # padding
}
conll_sorted_ids_dct = {
    'perm0': [0, 1, 2, 3],  # normal PER → LOC → ORG → MISC
    'perm1': [0, 3, 1, 2],  # perm
    'perm2': [1, 0, 2, 3],  # perm
    'perm3': [1, 2, 3, 0],  # perm
    'perm4': [2, 1, 3, 0],  # perm
    'perm5': [2, 3, 0, 1],  # perm
    'perm6': [3, 0, 1, 2],  # perm
    'perm7': [3, 2, 0, 1],  # perm
}
fewnerd_sorted_ids_dct = {
    'perm0': [0, 1, 2, 3, 4, 5, 6, 7],  # normal LOC → PER → ORG → OTH → PROD → BUID → ART → EVET
    'perm1': [2, 4, 6, 7, 3, 1, 0, 5],  # perm   ORG → PROD → ART → EVET → OTH → PER → LOC → BUID
    'perm2': [4, 7, 3, 1, 6, 0, 5, 2],  # perm   PROD → EVET → OTH → PER → ART → LOC → BUID → ORG
    'perm3': [5, 3, 4, 1, 2, 0, 6, 7],  # perm   BUID → OTH → PROD → PER → ORG → LOC → ART → EVET
}

if __name__ == "__main__":
    from rich.logging import RichHandler

    # handlers = [logging.StreamHandler(sys.stdout)]
    file_handler = logging.FileHandler(f"training.log", mode='a')
    file_handler.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s] %(message)s',
                                                datefmt="%Y-%m-%d %H:%M:%S"))
    rich_handle = RichHandler(rich_tracebacks=True, )
    rich_handle.setFormatter(logging.Formatter(fmt='%(message)s',
                                               datefmt="%Y-%m-%d %H:%M:%S"))
    logging.basicConfig(level=logging.INFO,
                        format="%(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        handlers=[rich_handle, file_handler])

    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser("clner")
    parser.add_argument('--info', help='information to distinguish model.', default='')
    parser.add_argument('--gpu', default='-2', type=str)  # '-1' use cpu, '-2' auto assign gpu
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--split_seed', default=2021, type=int)
    parser.add_argument('--ckpt_dir', default=Path('model_ckpt'), type=Path)
    parser.add_argument('--m', default='mlm', choices=['mlm', 'spankl', 'ext', 'add'], type=str, help='type of model')  # spankl addner extendner
    parser.add_argument('--perm', help='permutation.', default='perm0', type=str)
    parser.add_argument('--std', help='ways of sampling the training dataset', default='all', choices=['all', 'filter'],type=str)
    parser.add_argument('--pretrain_mode', default='fine_tuning', choices=['fine_tuning', 'feature_based'], type=str)
    parser.add_argument('--corpus', default='onto', choices=['onto', 'fewnerd', 'onto_fs', 'conll'], type=str)
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument('--batch_size', default=[
        16,
        24,  # fewnerd
        32,  # onto
        48
    ][2], type=int)

    parser.add_argument('--non_cl', default=False, type=utils.str2bool)  # use non-CL-complete or not
    parser.add_argument('--setup', default='split', choices=['split', 'filter'], type=str)  # Synthetic Setup of Training Set

    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--t0_num_epochs', default=5, type=int)
    parser.add_argument('--bert_lr', default=1e-4, type=float)  # finetune bert lr
    parser.add_argument('--lr', default=0.001, type=float)  # other subnetworks fast lr
    parser.add_argument('--warmup_step', default=0, type=int)

    parser.add_argument('--use_distill', default=True, type=utils.str2bool)
    parser.add_argument('--use_best_dev', default=False, type=utils.str2bool)
    parser.add_argument('--also_Test_Filter', default=True, type=utils.str2bool)
    parser.add_argument('--quick_test', default=False, action='store_true', help='use a partial data for quick verification of the code')
    parser.add_argument('--distill_dev', default=False, type=utils.str2bool)  # try to also distill Dev set to imporve the measure accuracy, not boost much, not use in published paper.

    # rz+
    parser.add_argument(
        "--label_map_path",
        default=None,
        type=str,
        help="label map path",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    # rz+

    # Below params is not applicable for this published paper, as experimental setup during research.
    # we have already tried the Gumbel Binary Gated mechanism by a learning task embedding.
    # It's effective in small featured based model but not much increasing especiall in large finetuned model.
    # May be left for future work.
    parser.add_argument('--use_task_embed', default=False, type=utils.str2bool)
    parser.add_argument('--use_gumbel_softmax', default=False, type=utils.str2bool)
    # A experimental setting to verify the Standard Supervised Full Data performance. != cl or non_cl
    parser.add_argument('--all_tasks', default=False, type=utils.str2bool)  # all tasks (Standard Supervised Full Data)
    args = parser.parse_args()

    # args.bert_model_dir = ['huggingface_model_resource/bert-base-cased', 'huggingface_model_resource/bert-large-cased'][1]
    args.bert_model_dir = ['PLM/bert-base-cased', 'PLM/bert-large-cased'][0]
    args.enc_dropout = [0.1, 0.2][0]
    if args.corpus == 'onto' or args.corpus == 'onto_fs':
        args.perm_ids = onto_sorted_ids_dct[args.perm]
        args.batch_size = 2
        args.num_epochs = 10
        args.t0_num_epochs = 5
    if args.corpus == 'fewnerd':
        args.perm_ids = fewnerd_sorted_ids_dct[args.perm]
        args.batch_size = 24
        args.num_epochs = 5
        args.t0_num_epochs = 5
    if args.corpus == 'conll':
        args.perm_ids = conll_sorted_ids_dct[args.perm]
        args.batch_size = 2
        args.num_epochs = 10
        args.t0_num_epochs = 1

    logger.info(utils.header_format("Starting", sep='='))

    metric_mlm = load_metric("./seqeval_metric.py")

    """ train """
    train_main(args)
    exit(0)

    """ eval """
    # eval_ckpt(args, existed_model_ckpt_dir='onto-0-2022-07-21_10-02-57-1824-spankl_split_perm0')
    # exit(0)

    """print metric used in the paper, see print_cl_metric.py"""
    # existed_model_ckpt_dir = 'model_ckpt/onto-0-2022-07-21_10-02-57-1824-spankl_split_perm0'
    # simply_print_cl_metric(existed_model_ckpt_dir)
