# !/usr/bin/env python
# coding=utf-8
"""
author: yonas
"""
import argparse

import torch
import numpy as np
from pathlib import Path
from transformers import BertTokenizer, AutoTokenizer, RobertaTokenizer
import datautils as utils
from datautils import NerExample, Any2Id
import time, copy
import random
import ipdb
from types import MethodType

try:
    from prefetch_generator import BackgroundGenerator  # prefetch-generator


    class DataLoaderX(torch.utils.data.DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
except:
    pass


def is_whitespace(char):
    if char in [' ', '\n', '\t', '\r']:
        return True
    return False


class CharTokenizer(AutoTokenizer):  # 为了适配robert-large 的vocab.json
# class CharTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(CharTokenizer, self).__init__(*args, **kwargs)
        self.fast_get_vocab = self.vocab

    def char_tokenize(self, text, **kwargs):
        """tokenize by char"""
        token_list = []
        for c in text:
            if c in self.vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(self.unk_token)
        return token_list


def get_char_tokenize_fn(tokenizer):
    vocab = tokenizer.vocab  # AutoTokenizer.vocab会非常耗时，先一次获取?

    # vocab = tokenizer.get_vocab()
    def char_tokenize_fn(text):
        token_list = []
        # time0 = time.time()
        for c in text:
            if c in vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(tokenizer.unk_token)
        # print('a', time.time() - time0)
        return token_list

    return char_tokenize_fn
# index = self._tokenizer.token_to_id(token)
# if index is None:
#     return self.unk_token_id


class NerDataReader:
    def __init__(self, tokenizer_path, max_len, ori_label_token_map, ent_file_or_ent_lst, loss_type=None, args=None):
        self.tokenizer_path = tokenizer_path
        if 'roberta' in tokenizer_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, add_prefix_space=True)  # JAPAN -> "ĠJ", "AP", "AN"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, do_lower_case=False)  # rz+ 必须设置do_lower_case=False
        self.char_tokenize_fn = get_char_tokenize_fn(self.tokenizer)  # used to handle ZH
        self.max_len = max_len
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self.loss_type = loss_type

        self.char2id = Any2Id(exist_dict=self.tokenizer.vocab)

        # if is softmax, should add 'O'
        if isinstance(ent_file_or_ent_lst, list):
            self.ent2id = Any2Id(exist_dict={e: i for i, e in enumerate(ent_file_or_ent_lst)})
        else:
            self.ent2id = Any2Id.from_file(ent_file_or_ent_lst, use_line_no=True)

        # tag2id = {'[PAD]': 0, 'O': 1}
        tag2id = {'O': 0}  # use O as pad simultaneously
        for ent in self.ent2id:
            if ent not in tag2id:
                tag2id[f'B-{ent}'] = len(tag2id)
                tag2id[f'I-{ent}'] = len(tag2id)

        self.tag2id = Any2Id(exist_dict=tag2id)

        self.id2char = self.char2id.get_reverse()
        self.id2ent = self.ent2id.get_reverse()
        self.id2tag = self.tag2id.get_reverse()
        self.ori_label_token_map = ori_label_token_map   # rz+

        # self.args = args
        self.args = argparse.Namespace()
        self.args.pretrain_mode = 'feature_based'
        self.args.pretrain_mode = 'fine_tuning'
        self.args.label_schema = 'IO'  # rz+
        self.args.prompt_seed = 0  # rz+
        self.args.use_refine_mask = False

    def create_mask(self, start, end, context_size):
        mask = torch.zeros(context_size, dtype=torch.bool)
        mask[start:end] = 1
        return mask

    def post_process(self, exm: NerExample, lang='ENG', train=True, arch='seq', loss_type='sigmoid'):
        if train:
            if not hasattr(exm, 'train_cache'):  # 如果exm对象不具有名为train_cache的属性
                if lang == 'ENG':  # ENG means having sub_tokens

                    ori_label_token_map = self.ori_label_token_map
                    prompt_words = []
                    prompt_tokens = []
                    prompt_tag_lst = []
                    # 确定要添加的类别数量
                    num_classes_to_add = exm.task_id + 1
                    num_entities_per_class = 2  # TODO 每个类别添加的实体模板数量

                    for i in range(num_classes_to_add):
                        entity_label = self.id2ent.any2id[i]
                        label_key = 'I-' + entity_label
                        if label_key in ori_label_token_map:
                            available_entities = ori_label_token_map[label_key][:]  # 复制一份列表
                            random.seed(self.args.prompt_seed)   # 这个种子确保了每次生成的演示模板的一致性，性能会好（Good Examples Make A Faster这篇论文的结论）
                            for _ in range(num_entities_per_class):
                                if not available_entities:
                                    break  # 如果没有更多的实体可选，则退出循环
                                entity = random.choice(available_entities)
                                available_entities.remove(entity)  # 确保每个实体只被选中一次
                                prompt_words.append(entity)
                                entity_tokens = self.tokenizer.tokenize(" " + entity)
                                for idx, sub_token in enumerate(entity_tokens):
                                    prompt_tokens.append(sub_token)
                                    # 对于第一个子标记使用 B- 标签，其余使用 I- 标签
                                    if idx == 0:
                                        prompt_tag_lst.append(label_key.replace('I-', 'B-'))
                                    else:
                                        prompt_tag_lst.append(label_key)  # 使用原始 I- 标签

                                prompt_words.append("belongs")
                                prompt_tokens.append("belongs")
                                prompt_tag_lst.append("O")
                                prompt_words.append("to")
                                prompt_tokens.append("to")
                                prompt_tag_lst.append("O")
                                prompt_words.append(label_key)
                                prompt_tokens.append(label_key)
                                prompt_tag_lst.append(label_key.replace('I-', 'B-'))
                                prompt_words.append(".")
                                prompt_tokens.append(".")
                                prompt_tag_lst.append("O")
                            # prompt_tokens.append(self.tokenizer.sep_token)
                    # # 加上例句
                    # example_sentences = {
                    #     "MISC": {
                    #         "words": ['The', 'Palestinians', 'are', 'an', 'ethnonational', 'group', 'descending', 'from', 'peoples', 'who', 'have', 'lived', 'in', 'Palestine', 'over', 'the', 'centuries', '.'],
                    #         "tokens": ['The', 'Palestinians', 'are', 'an', 'et', '##hn', '##ona', '##tional', 'group', 'descending', 'from', 'peoples', 'who', 'have', 'lived', 'in', 'Palestine', 'over', 'the', 'centuries', '.'],
                    #         "labels": ['B-MISC', 'I-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']
                    #     },
                    #     "PER": {
                    #         "words": ['John', 'Fitzgerald', 'Kennedy', 'was', 'the', '35th', 'president', 'of', 'the', 'United', 'States', ',', 'serving', 'from', 'January', '1961', 'until', 'his', 'assassination', 'in', 'November', '1963', '.'],
                    #         "tokens": ['John', 'Fitzgerald', 'Kennedy', 'was', 'the', '35th', 'president', 'of', 'the', 'United', 'States', ',', 'serving', 'from', 'January', '1961', 'until', 'his', 'assassination', 'in', 'November', '1963', '.'],
                    #         "labels": ['B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
                    #     },
                    #     "LOC": {
                    #         "words": ['England', 'is', 'a', 'country', 'that', 'is', 'part', 'of', 'the', 'United', 'Kingdom', ',', 'sharing', 'borders', 'with', 'Scotland', 'to', 'the', 'north', 'and', 'Wales', 'to', 'the', 'west', '.'],
                    #         "tokens": ['England', 'is', 'a', 'country', 'that', 'is', 'part', 'of', 'the', 'United', 'Kingdom', ',', 'sharing', 'borders', 'with', 'Scotland', 'to', 'the', 'north', 'and', 'Wales', 'to', 'the', 'west', '.'],
                    #         "labels": ['B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']
                    #     },
                    #     "ORG": {
                    #         "words": ['The', 'Communist', 'Party', 'of', 'China', 'is', 'the', 'founding', 'and', 'ruling', 'party', 'of', 'the', 'People', "'s", 'Republic', 'of', 'China', '.'],
                    #         "tokens": ['The', 'Communist', 'Party', 'of', 'China', 'is', 'the', 'founding', 'and', 'ruling', 'party', 'of', 'the', 'People', "'", 's', 'Republic', 'of', 'China', '.'],
                    #         "labels": ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O']
                    #     }
                    # }
                    # example_sentences = {
                    #     "MISC": {
                    #         "words": ['Dutch', 'state', 'raises', 'tap', 'sale', 'price', 'to', '99.95', '.', 'Fifty',
                    #                   'Russians', 'die', 'in', 'clash', 'with', 'rebels-Interfax', '.'
                    #                   ],
                    #         "tokens": ['Dutch', 'state', 'raises', 'tap', 'sale', 'price', 'to', '99', '.', '95', '.',
                    #                    'Fifty', 'Russians', 'die', 'in', 'clash', 'with', 'rebels', '-', 'Inter',
                    #                    '##fa',
                    #                    '##x', '.'
                    #                    ],
                    #         "labels": ['B-MISC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O',
                    #                    'O',
                    #                    'O', 'O'
                    #                    ]
                    #     },
                    #     "PER": {
                    #         "words": ['Second-ranked', 'Austrian', 'Thomas', 'Muster', ',', 'who', 'was', 'seeded',
                    #                   'third',
                    #                   ',', 'did', 'not', 'have', 'the', 'luck', 'of', 'the', 'draw', 'with', 'him', '.',
                    #                   'Five', 'other', 'people', 'have', 'been', 'arrested', 'including', 'Dutroux',
                    #                   "'s",
                    #                   'second', 'wife', 'Michelle', 'Martin', ',', 'charged', 'as', 'an', 'accomplice',
                    #                   '.'
                    #                   ],
                    #         "tokens": ['Second', '-', 'ranked', 'Austrian', 'Thomas', 'Must', '##er', ',', 'who', 'was',
                    #                    'seeded', 'third', ',', 'did', 'not', 'have', 'the', 'luck', 'of', 'the', 'draw',
                    #                    'with', 'him', '.', 'Five', 'other', 'people', 'have', 'been', 'arrested',
                    #                    'including',
                    #                    'Du', '##tro', '##ux', "'", 's', 'second', 'wife', 'Michelle', 'Martin', ',',
                    #                    'charged', 'as', 'an', 'a', '##cco', '##mp', '##lice', '.'
                    #                    ],
                    #         "labels": ['O', 'B-MISC', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                    #                    'O', 'O',
                    #                    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'O',
                    #                    'O',
                    #                    'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O'
                    #                    ]
                    #     },
                    #     "LOC": {
                    #         "words": ['Only', 'France', 'and', 'Britain', 'backed', 'Fischler', "'s", 'proposal', '.',
                    #                   'Iraq', "'s", 'Saddam', 'meets', 'Russia', "'s", 'Zhirinovsky', '.'
                    #                   ],
                    #         "tokens": ['Only', 'France', 'and', 'Britain', 'backed', 'Fi', '##sch', '##ler', "'", 's',
                    #                    'proposal', '.', 'Iraq', "'", 's', 'Saddam', 'meets', 'Russia', "'", 's', 'Z',
                    #                    '##hir',
                    #                    '##ino', '##vsky', '.'
                    #                    ],
                    #         "labels": ['O', 'B-LOC', 'O', 'B-LOC', 'O', 'B-PER', 'O', 'O', 'O', 'B-LOC', 'O', 'B-PER',
                    #                    'O',
                    #                    'B-LOC', 'O', 'B-PER', 'O'
                    #                    ]
                    #     },
                    #     "ORG": {
                    #         "words": ['He', 'said', 'further', 'scientific', 'study', 'was', 'required', 'and', 'if',
                    #                   'it',
                    #                   'was', 'found', 'that', 'action', 'was', 'needed', 'it', 'should', 'be', 'taken',
                    #                   'by',
                    #                   'the', 'European', 'Union', '.', 'The', 'Bank', 'of', 'Finland', 'earlier',
                    #                   'estimated', 'the', 'April', 'trade', 'surplus', 'at', '3.2', 'billion', 'markka',
                    #                   'with', 'exports', 'projected', 'at', '14.5', 'billion', 'and', 'imports', 'at',
                    #                   '11.3', 'billion', '.'
                    #                   ],
                    #         "tokens": ['He', 'said', 'further', 'scientific', 'study', 'was', 'required', 'and', 'if',
                    #                    'it',
                    #                    'was', 'found', 'that', 'action', 'was', 'needed', 'it', 'should', 'be', 'taken',
                    #                    'by',
                    #                    'the', 'European', 'Union', '.', 'The', 'Bank', 'of', 'Finland', 'earlier',
                    #                    'estimated', 'the', 'April', 'trade', 'surplus', 'at', '3', '.', '2', 'billion',
                    #                    'mark', '##ka', 'with', 'exports', 'projected', 'at', '14', '.', '5', 'billion',
                    #                    'and',
                    #                    'imports', 'at', '11', '.', '3', 'billion', '.'
                    #                    ],
                    #         "labels": ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                    #                    'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'B-ORG', 'I-ORG',
                    #                    'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
                    #     }
                    # }
                    # if exm.task_id == 1:
                    #     prompt_words.extend(example_sentences[self.id2ent[0]]['words'])
                    #     prompt_tokens.extend(example_sentences[self.id2ent[0]]['tokens'])
                    #     prompt_tag_lst.extend(example_sentences[self.id2ent[0]]['labels'])
                    # elif exm.task_id == 2:
                    #     prompt_words.extend(example_sentences[self.id2ent[0]]['words'])
                    #     prompt_tokens.extend(example_sentences[self.id2ent[0]]['tokens'])
                    #     prompt_tag_lst.extend(example_sentences[self.id2ent[0]]['labels'])
                    #     prompt_words.extend(example_sentences[self.id2ent[1]]['words'])
                    #     prompt_tokens.extend(example_sentences[self.id2ent[1]]['tokens'])
                    #     prompt_tag_lst.extend(example_sentences[self.id2ent[1]]['labels'])
                    # elif exm.task_id == 3:
                    #     prompt_words.extend(example_sentences[self.id2ent[0]]['words'])
                    #     prompt_tokens.extend(example_sentences[self.id2ent[0]]['tokens'])
                    #     prompt_tag_lst.extend(example_sentences[self.id2ent[0]]['labels'])
                    #     prompt_words.extend(example_sentences[self.id2ent[1]]['words'])
                    #     prompt_tokens.extend(example_sentences[self.id2ent[1]]['tokens'])
                    #     prompt_tag_lst.extend(example_sentences[self.id2ent[1]]['labels'])
                    #     prompt_words.extend(example_sentences[self.id2ent[2]]['words'])
                    #     prompt_tokens.extend(example_sentences[self.id2ent[2]]['tokens'])
                    #     prompt_tag_lst.extend(example_sentences[self.id2ent[2]]['labels'])
                    # # 加上例句
                    # ori_label_token_map = self.ori_label_token_map
                    # prompt_words = []
                    # prompt_tokens = []
                    # prompt_tag_lst = []
                    # for entity_label in ori_label_token_map:
                    #     random.seed(self.args.prompt_seed)
                    #     entity = random.choice(ori_label_token_map[entity_label])
                    #     prompt_words.append(entity)
                    #     entity_tokens = self.tokenizer.tokenize(" " + entity)
                    #     for idx, sub_token in enumerate(entity_tokens):
                    #         prompt_tokens.append(sub_token)
                    #         # 对于第一个子标记使用 B- 标签，其余使用 I- 标签
                    #         if idx == 0:
                    #             prompt_tag_lst.append(entity_label.replace('I-', 'B-'))
                    #         else:
                    #             prompt_tag_lst.append(entity_label)  # 使用原始 I- 标签
                    #
                    #     prompt_words.append("is")
                    #     prompt_tokens.append("is")
                    #     prompt_tag_lst.append("O")
                    #     prompt_words.append(entity_label)
                    #     prompt_tokens.append(entity_label)
                    #     prompt_tag_lst.append(entity_label.replace('I-', 'B-'))
                    #     prompt_words.append(".")
                    #     prompt_tokens.append(".")
                    #     prompt_tag_lst.append("O")
                    #     # prompt_tokens.append(self.tokenizer.sep_token)

                    # prompt_words.extend(["W", "is", "not", "an", "entity", ".", "NFL", "is", "not", "an", "entity", ".", "123", "is", "not", "an", "entity", "."])
                    # prompt_tokens.extend(["W", "is", "not", "an", "entity", ".", "NFL", "is", "not", "an", "entity", ".", "123", "is", "not", "an", "entity", "."])
                    # prompt_tag_lst.extend(["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"])
                    # prompt_words = ['('] + prompt_words + [')']
                    # prompt_tokens = ['('] + prompt_tokens + [')']
                    # prompt_tag_lst = ['O'] + prompt_tag_lst + ['O']
                    input_ids = self.tokenizer.convert_tokens_to_ids(exm.bert_tok_char_lst)
                    prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
                    # input_encoding = self.tokenizer(exm.text)  # tokenizer()方法得到的结果可以调用.word_ids()
                    input_words = exm.char_lst + prompt_words
                    # input_encoding = self.tokenizer(exm.char_lst, truncation=True,
                    input_encoding = self.tokenizer(input_words, truncation=True,
                                                    is_split_into_words=True)  # 知识点：tokenizer()方法传入不同的类型（str或list），输出的.word_ids()结果不一样;并且一定要加 truncation=True, is_split_into_words=True参数
                    input_encoding_wo_st = self.tokenizer(input_words, truncation=True,
                                                    is_split_into_words=True, add_special_tokens=False)
                elif lang == 'ZH':  # split by each char
                    input_ids = self.tokenizer.convert_tokens_to_ids(self.char_tokenize_fn(exm.char_lst))
                    input_encoding = self.tokenizer(exm.char_lst)

                # input_ids = [self.cls_id] + input_ids + [self.sep_id]
                input_ids = [self.cls_id] + input_ids + prompt_ids + [self.sep_id]
                if len(input_ids) > 512:
                    print('超出长度！！！！！！！！！')  # TODO 改
                exm.train_cache = dict(input_ids=input_ids, len=len(input_ids))
                if lang == 'ENG':
                    # rz+ 源代码使用的exm.ori_2_tok，我将其改为加了prompt之后的ori_2_tok
                    tokenized_word_ids = input_encoding_wo_st.word_ids()  # [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                    ori_2_tok = [-1] * len(input_words)  # ori_char_lst[i] -> tok_char_lst[ori_char_lst_2_tok_char_lst[i]]
                    for idx, word_id in enumerate(tokenized_word_ids):
                        if ori_2_tok[word_id] == -1:
                            ori_2_tok[word_id] = idx
                    # exm.train_cache.update(ori_len=len(exm.char_lst), ori_2_tok=exm.ori_2_tok, wo_pad_len=len(input_ids))
                    exm.train_cache.update(ori_len=len(input_words), ori_2_tok=ori_2_tok, wo_pad_len=len(input_ids))
                if train:
                    if arch == 'seq':
                        tag_lst = NerExample.to_tag_lst(exm.char_lst, exm.ent_dct)
                        all_tag_lst = tag_lst + prompt_tag_lst
                        tag_ids = [self.tag2id[tag] for tag in all_tag_lst]
                        exm.train_cache.update(tag_ids=tag_ids)

                        # add MLM 标签
                        label_token_map = {item: item for item in self.ori_label_token_map}
                        label_token_to_id = {label: self.tokenizer.convert_tokens_to_ids(label_token) for
                                             label, label_token in label_token_map.items()}

                        ent_size = len(self.ent2id)
                        word_ids = input_encoding.word_ids()

                        ### FIT论文中的pooling
                        word_mask = []
                        # for i, t in enumerate(exm.char_lst):
                        for i, t in enumerate(input_words):
                            if i in word_ids:
                                indices = [index for index, value in enumerate(word_ids) if value == i]
                                index_range = [indices[0], indices[-1]+1]
                                word_mask.append(self.create_mask(*index_range, len(input_ids)))
                            else:
                                raise ValueError(f"For i={i}, the value is not in word_ids")
                        word_mask = torch.stack(word_mask)

                        previous_word_idx = None
                        label_ids = []
                        targe_token_onehot = np.tile(np.array(input_ids, dtype='float32')[:, None], ent_size)
                        targe_token_onehot[0, :] = -100  # [CLS]
                        targe_token_onehot[-1, :] = -100  # [SEP]
                        target_token = []
                        for i, (input_idx, word_idx) in enumerate(zip(input_ids, word_ids)):
                            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                            # ignored in the loss function.

                            if word_idx is None:
                                target_token.append(-100)
                                label_ids.append(-100)
                            # Set target token for the first token of each word.
                            elif word_idx != previous_word_idx:  # 若不是 ##ed 之类的子词
                                # label_ids.append(self.tag2id.any2id[tag_lst[word_idx]])
                                label_ids.append(self.tag2id.any2id[all_tag_lst[word_idx]])
                                if self.args.label_schema == "IO" and all_tag_lst[word_idx] != "O":  # 这一步是将BIO变为IO，将所有B-换为I-。
                                    all_tag_lst[word_idx] = "I-" + all_tag_lst[word_idx][2:]

                                if all_tag_lst[word_idx] != 'O':  # 若为实体标签
                                    targe_token_onehot[i, self.ent2id.any2id[all_tag_lst[word_ids[i]][2:]]] = label_token_to_id[all_tag_lst[word_idx]]
                                    target_token.append(label_token_to_id[all_tag_lst[word_idx]])
                                else:
                                    target_token.append(input_idx)
                                # target_tokens.append()

                            # Set target token for other tokens of each word.
                            else:  # 若是 ##ed 之类的子词
                                label_ids.append(-100)
                                if self.args.label_schema == "IO" and all_tag_lst[word_idx] != "O":
                                    all_tag_lst[word_idx] = "I-" + all_tag_lst[word_idx][2:]

                                if all_tag_lst[word_idx] != 'O':
                                    # Set the same target token for each tokens.
                                    targe_token_onehot[i, self.ent2id.any2id[all_tag_lst[word_ids[i]][2:]]] = label_token_to_id[all_tag_lst[word_idx]]
                                    target_token.append(label_token_to_id[all_tag_lst[word_idx]])
                                else:
                                    target_token.append(input_idx)
                            previous_word_idx = word_idx
                        exm.train_cache.update(tag_token_onehot=targe_token_onehot, tag_token=target_token,
                                               gold_label_ids=label_ids, word_mask=word_mask)

                    elif arch == 'span':
                        assert loss_type in ['sigmoid', 'softmax']
                        ent_size = len(self.ent2id)
                        span_ner_tgt_lst = exm.get_span_level_ner_tgt_lst(
                            neg_symbol='O')  # 将原始句子枚举所有span，对每个span打标签，非实体为O（如原始句子12个token，则枚举78个span，对这78个打标签）
                        if loss_type == 'sigmoid':
                            # use one-hot
                            # 一般torch的向量都是float()而默认的numpy则是doble(float64)
                            span_tgt_onehot = np.zeros([len(span_ner_tgt_lst), ent_size],
                                                       dtype='float32')  # [num_spans, ent] (78,6)
                            for i, tag in enumerate(span_ner_tgt_lst):
                                if tag != 'O' and tag in self.ent2id:
                                    span_tgt_onehot[i][self.ent2id[tag]] = 1.
                            exm.train_cache.update(span_tgt=span_tgt_onehot)
                        elif loss_type == 'softmax':
                            span_tgt = [self.ent2id[e] for e in span_ner_tgt_lst]
                            exm.train_cache.update(span_tgt=span_tgt)
                    else:
                        raise NotImplementedError

            # other setting
            if hasattr(exm, 'distilled_span_ner_pred_lst'):
                num_spans, so_far_ent_size = exm.distilled_span_ner_pred_lst.shape
                distilled_span_ner_tgt_lst = copy.deepcopy(exm.train_cache['span_tgt'])  # [num_spans, ent]
                distilled_span_ner_tgt_lst[:, :so_far_ent_size] = exm.distilled_span_ner_pred_lst
                exm.train_cache['distilled_span_tgt'] = distilled_span_ner_tgt_lst
                delattr(exm, 'distilled_span_ner_pred_lst')

            if hasattr(exm, 'distilled_task_ent_output'):
                # exm.train_cache['distilled_task_ent_output'] = exm.distilled_task_ent_output
                distilled_task_ent_output = exm.distilled_task_ent_output[:len(exm.train_cache['input_ids'])]
                exm.train_cache['distilled_task_ent_output'] = distilled_task_ent_output
            else:
                if 'distilled_task_ent_output' in exm.train_cache:
                    exm.train_cache.pop('distilled_task_ent_output')
            # ipdb.set_trace()
            return dict(ner_exm=exm, **exm.train_cache)
        if not train:
            if not hasattr(exm, 'train_cache'):  # 如果exm对象不具有名为train_cache的属性
                if lang == 'ENG':  # ENG means having sub_tokens
                    # # TODO
                    # ori_label_token_map = {"I-ORG": ["National", "Corp", "News", "Inc", "Senate", "Court"],
                    #                        "I-PERSON": ["John", "David", "Peter", "Michael", "Robert", "James"],
                    #                        "I-GPE": ["US", "China", "United", "Beijing", "Israel", "Taiwan"],
                    #                        "I-DATE": ["Year", "December", "August", "July", "1940", "March"],
                    #                        "I-CARDINAL": ["Two", "four", "Three", "Hundred", "20", "8"],
                    #                        "I-NORP": ["Chinese", "Israeli", "Palestinians", "American", "Japanese",
                    #                                   "Palestinian"]}
                    # prompt_tokens = []
                    # for entity_label in ori_label_token_map:
                    #     entity = random.choice(ori_label_token_map[entity_label])
                    #     entity_tokens = self.tokenizer.tokenize(" " + entity)
                    #     for sub_token in entity_tokens:
                    #         prompt_tokens.append(sub_token)
                    #
                    #     prompt_tokens.append("is")
                    #     prompt_tokens.append(entity_label)
                    #     prompt_tokens.append(".")
                    #     prompt_tokens.append(self.tokenizer.sep_token)

                    input_ids = self.tokenizer.convert_tokens_to_ids(exm.bert_tok_char_lst)
                    # prompt_ids = self.tokenizer.convert_tokens_to_ids(prompt_tokens)
                    # input_encoding = self.tokenizer(exm.text)  # tokenizer()方法得到的结果可以调用.word_ids()
                    input_encoding = self.tokenizer(exm.char_lst, truncation=True,
                                                    is_split_into_words=True, )  # 知识点：tokenizer()方法传入不同的类型（str或list），输出的.word_ids()结果不一样;并且一定要加 truncation=True, is_split_into_words=True参数
                elif lang == 'ZH':  # split by each char
                    input_ids = self.tokenizer.convert_tokens_to_ids(self.char_tokenize_fn(exm.char_lst))
                    input_encoding = self.tokenizer(exm.char_lst)

                input_ids = [self.cls_id] + input_ids + [self.sep_id]
                # input_ids = [self.cls_id] + input_ids + [self.sep_id] + prompt_ids
                if len(input_ids) > 512:
                    print('超出长度！！！！！！！！！')
                exm.train_cache = dict(input_ids=input_ids, len=len(input_ids))
                if lang == 'ENG':
                    exm.train_cache.update(ori_len=len(exm.char_lst), ori_2_tok=exm.ori_2_tok,
                                           wo_pad_len=len(input_ids))
            # if train:
                if arch == 'seq':
                    tag_lst = NerExample.to_tag_lst(exm.char_lst, exm.ent_dct)
                    tag_ids = [self.tag2id[tag] for tag in tag_lst]
                    exm.train_cache.update(tag_ids=tag_ids)

                    # add MLM 标签
                    label_token_map = {item: item for item in self.ori_label_token_map}
                    label_token_to_id = {label: self.tokenizer.convert_tokens_to_ids(label_token) for
                                         label, label_token in label_token_map.items()}

                    ent_size = len(self.ent2id)
                    word_ids = input_encoding.word_ids()

                    ### FIT论文中的pooling
                    word_mask = []
                    for i, t in enumerate(exm.char_lst):
                        if i in word_ids:
                            indices = [index for index, value in enumerate(word_ids) if value == i]
                            index_range = [indices[0], indices[-1] + 1]
                            word_mask.append(self.create_mask(*index_range, len(input_ids)))
                        else:
                            raise ValueError(f"For i={i}, the value is not in word_ids")
                    word_mask = torch.stack(word_mask)

                    previous_word_idx = None
                    label_ids = []
                    targe_token_onehot = np.tile(np.array(input_ids, dtype='float32')[:, None], ent_size)
                    targe_token_onehot[0, :] = -100  # [CLS]
                    targe_token_onehot[-1, :] = -100  # [SEP]
                    target_token = []
                    for i, (input_idx, word_idx) in enumerate(zip(input_ids, word_ids)):
                        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                        # ignored in the loss function.

                        if word_idx is None:
                            target_token.append(-100)
                            label_ids.append(-100)
                        # Set target token for the first token of each word.
                        elif word_idx != previous_word_idx:  # 若不是 ##ed 之类的子词
                            label_ids.append(self.tag2id.any2id[tag_lst[word_idx]])
                            if self.args.label_schema == "IO" and tag_lst[
                                word_idx] != "O":  # 这一步是将BIO变为IO，将所有B-换为I-。
                                tag_lst[word_idx] = "I-" + tag_lst[word_idx][2:]

                            if tag_lst[word_idx] != 'O':  # 若为实体标签
                                targe_token_onehot[i, self.ent2id.any2id[tag_lst[word_ids[i]][2:]]] = \
                                    label_token_to_id[tag_lst[word_idx]]
                                target_token.append(label_token_to_id[tag_lst[word_idx]])
                            else:
                                target_token.append(input_idx)
                            # target_tokens.append()

                        # Set target token for other tokens of each word.
                        else:  # 若是 ##ed 之类的子词
                            label_ids.append(-100)
                            if self.args.label_schema == "IO" and tag_lst[word_idx] != "O":
                                tag_lst[word_idx] = "I-" + tag_lst[word_idx][2:]

                            if tag_lst[word_idx] != 'O':
                                # Set the same target token for each tokens.
                                targe_token_onehot[i, self.ent2id.any2id[tag_lst[word_ids[i]][2:]]] = \
                                    label_token_to_id[tag_lst[word_idx]]
                                target_token.append(label_token_to_id[tag_lst[word_idx]])
                            else:
                                target_token.append(input_idx)
                        previous_word_idx = word_idx
                    exm.train_cache.update(tag_token_onehot=targe_token_onehot, tag_token=target_token,
                                           gold_label_ids=label_ids, word_mask=word_mask)

                elif arch == 'span':
                    assert loss_type in ['sigmoid', 'softmax']
                    ent_size = len(self.ent2id)
                    span_ner_tgt_lst = exm.get_span_level_ner_tgt_lst(
                        neg_symbol='O')  # 将原始句子枚举所有span，对每个span打标签，非实体为O（如原始句子12个token，则枚举78个span，对这78个打标签）
                    if loss_type == 'sigmoid':
                        # use one-hot
                        # 一般torch的向量都是float()而默认的numpy则是doble(float64)
                        span_tgt_onehot = np.zeros([len(span_ner_tgt_lst), ent_size],
                                                   dtype='float32')  # [num_spans, ent] (78,6)
                        for i, tag in enumerate(span_ner_tgt_lst):
                            if tag != 'O' and tag in self.ent2id:
                                span_tgt_onehot[i][self.ent2id[tag]] = 1.
                        exm.train_cache.update(span_tgt=span_tgt_onehot)
                    elif loss_type == 'softmax':
                        span_tgt = [self.ent2id[e] for e in span_ner_tgt_lst]
                        exm.train_cache.update(span_tgt=span_tgt)
                else:
                    raise NotImplementedError

            # other setting
            if hasattr(exm, 'distilled_span_ner_pred_lst'):
                num_spans, so_far_ent_size = exm.distilled_span_ner_pred_lst.shape
                distilled_span_ner_tgt_lst = copy.deepcopy(exm.train_cache['span_tgt'])  # [num_spans, ent]
                distilled_span_ner_tgt_lst[:, :so_far_ent_size] = exm.distilled_span_ner_pred_lst
                exm.train_cache['distilled_span_tgt'] = distilled_span_ner_tgt_lst
                delattr(exm, 'distilled_span_ner_pred_lst')

            if hasattr(exm, 'distilled_task_ent_output'):
                exm.train_cache['distilled_task_ent_output'] = exm.distilled_task_ent_output
            else:
                if 'distilled_task_ent_output' in exm.train_cache:
                    exm.train_cache.pop('distilled_task_ent_output')
            # ipdb.set_trace()
            return dict(ner_exm=exm, **exm.train_cache)

    def get_batcher_fn(self, gpu=False, device=None, arch='span'):

        def tensorize(array, dtype='int'):
            if isinstance(array, np.ndarray):
                ret = torch.from_numpy(array)
            elif isinstance(array, torch.Tensor):
                ret = array
            else:  # list
                #  Creating a tensor from a list of numpy.ndarrays is extremely slow.
                #  Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
                if dtype == 'int':
                    ret = torch.LongTensor(array)
                elif dtype == 'float':
                    ret = torch.FloatTensor(array)
                elif dtype == 'double':
                    ret = torch.DoubleTensor(array)
                else:
                    raise NotImplementedError
            if gpu:
                if device is not None:
                    ret = ret.to(device)
                else:
                    ret = ret.cuda()
            return ret

        def span_batcher(batch_e):
            max_len = max(e['len'] for e in batch_e)  # length after bert tokenized, i.e. the longer sub-word
            batch_input_ids = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_seq_len = []
            batch_span_tgt_lst = []  # list of [num_spans, ent]
            batch_ner_exm = []

            batch_ori_seq_len = []
            batch_ori_2_tok = []
            if 'ori_len' in batch_e[
                0]:  # ori_len is the raw len, especially in ENG using tokenizer to split into longer subtokens
                ori_max_len = max(e['ori_len'] for e in batch_e)  # length before bert tokenized: shorter list

            if self.args.use_refine_mask:
                batch_refine_mask = np.zeros([len(batch_e), ori_max_len, ori_max_len])  # 0113
            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = []  # container for feature-based pt

            batch_span_tgt_lst_distilled = []  # list of [num_spans, ent]

            for bdx, e in enumerate(batch_e):
                batch_seq_len.append(e['len'] - 2)  # 去除cls和sep后的长度
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])

                if 'ori_len' in batch_e[0]:  # ENG
                    batch_ori_seq_len.append(e['ori_len'])
                    batch_ori_2_tok.append(e['ori_2_tok'] + [0] * (ori_max_len - e['ori_len']))

                if self.args.pretrain_mode == 'feature_based':  # feature-based pt
                    if hasattr(e['ner_exm'], 'pt'):
                        assert e['ner_exm'].pt.shape[0] == e['ori_len']
                        batch_input_pts.append(e['ner_exm'].pt)

                if 'span_tgt' in e:
                    batch_span_tgt_lst.append(tensorize(e['span_tgt']))  # list of [num_spans, ent]
                    if self.args.use_refine_mask:
                        batch_refine_mask[bdx, :e['ori_len'], :e['ori_len']] = e['ner_exm'].refine_mask  # 0113

                if 'distilled_span_tgt' in e:
                    batch_span_tgt_lst_distilled.append(
                        tensorize(e['distilled_span_tgt']))  # list of [num_spans, ent]

            if 'ori_len' not in batch_e[0]:  # ZH
                batch_ori_seq_len = batch_seq_len  # 方便兼容ZH时也能使用ori_seq_len

            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = torch.nn.utils.rnn.pad_sequence(batch_input_pts, batch_first=True,
                                                                  padding_value=0.)  # [b,len,1024]

            if batch_span_tgt_lst:
                batch_span_tgt = torch.cat(batch_span_tgt_lst, dim=0)  # [bsz*num_spans, ent]
            else:
                batch_span_tgt = None

            if batch_span_tgt_lst_distilled:
                batch_span_tgt_distilled = torch.cat(batch_span_tgt_lst_distilled, dim=0)
            else:
                batch_span_tgt_distilled = None

            return {
                'input_ids': tensorize(batch_input_ids),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'seq_len': tensorize(batch_seq_len),
                'batch_ner_exm': batch_ner_exm,

                'ori_seq_len': tensorize(batch_ori_seq_len),
                'batch_ori_2_tok': tensorize(batch_ori_2_tok),

                'batch_span_tgt': batch_span_tgt,
                'batch_span_tgt_lst': batch_span_tgt_lst,

                'batch_input_pts': tensorize(
                    batch_input_pts) if self.args.pretrain_mode == 'feature_based' else None,

                'batch_refine_mask': tensorize(batch_refine_mask) if self.args.use_refine_mask else None,

                'batch_span_tgt_distilled': batch_span_tgt_distilled,
                'batch_span_tgt_lst_distilled': batch_span_tgt_lst_distilled
            }

        def seq_batcher(batch_e):
            max_len = max(e['len'] for e in batch_e)
            batch_input_ids = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_seq_len = []
            batch_tag_ids = []
            batch_tag_token = []  # rz+
            batch_tag_token_onehot_lst = []  # rz+
            batch_word_mask_lst = []  # rz+
            batch_gold_label_ids = []  # rz+
            batch_ner_exm = []

            batch_ori_seq_len = []
            batch_wo_pad_len =[]
            batch_ori_2_tok = []
            if 'ori_len' in batch_e[0]:  # ori_len is the raw len, especially in ENG using tokenizer to split into longer subtokens
                ori_max_len = max(e['ori_len'] for e in batch_e)  # length before bert tokenized: shorter list

            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = []  # container for feature-based pt

            batch_distilled_task_ent_output = []

            for e in batch_e:
                batch_seq_len.append(e['len'] - 2)
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])

                if 'ori_len' in batch_e[0]:  # ENG
                    batch_ori_seq_len.append(e['ori_len'])
                    batch_ori_2_tok.append(e['ori_2_tok'] + [0] * (ori_max_len - e['ori_len']))

                if 'wo_pad_len' in batch_e[0]:  # ENG
                    batch_wo_pad_len.append(e['wo_pad_len'])

                if self.args.pretrain_mode == 'feature_based':  # feature-based pt
                    if hasattr(e['ner_exm'], 'pt'):
                        assert e['ner_exm'].pt.shape[0] == e['ori_len']
                    batch_input_pts.append(e['ner_exm'].pt)

                if 'tag_ids' in e:
                    batch_tag_ids.append(tensorize(e['tag_ids']))  # list of [len]

                if 'tag_token' in e:
                    batch_tag_token.append(tensorize(e['tag_token']))  # rz+

                if 'tag_token_onehot' in e:
                    batch_tag_token_onehot_lst.append(tensorize(e['tag_token_onehot']))  # rz+

                if 'word_mask' in e:
                    batch_word_mask_lst.append(tensorize(e['word_mask']))  # rz+

                if 'gold_label_ids' in e:
                    batch_gold_label_ids.append(tensorize(e['gold_label_ids']))  # rz+

                if 'distilled_task_ent_output' in e:  # logits蒸馏行不通
                    batch_distilled_task_ent_output.append(
                        tensorize(e['distilled_task_ent_output']))  # list of [len,ent]

                # if 'distilled_task_ent_output' in e:   # 改为自训练
                #     batch_distilled_task_ent_output.append(e['distilled_task_ent_output'])  # list of [len,ent]

            if 'ori_len' not in batch_e[0]:  # ZH
                batch_ori_seq_len = batch_seq_len  # 方便兼容ZH时也能使用ori_seq_len

            if self.args.pretrain_mode == 'feature_based':
                batch_input_pts = torch.nn.utils.rnn.pad_sequence(batch_input_pts, batch_first=True,
                                                                  padding_value=0.)  # [b,len,1024]

            if batch_tag_ids:
                if '[PAD]' in self.tag2id:
                    padding_value = self.tag2id['[PAD]']  # 补PAD 当tag2id有pad时
                else:
                    padding_value = self.tag2id['O']  # 补O
                batch_tag_ids = torch.nn.utils.rnn.pad_sequence(batch_tag_ids, batch_first=True,
                                                                padding_value=padding_value)  # [b,len]  # 补O
            else:
                batch_tag_ids = None

            if batch_tag_token:  # rz+
                if '[PAD]' in self.tag2id:
                    padding_value = self.tag2id['[PAD]']  # 补PAD 当tag2id有pad时
                else:
                    padding_value = -100  # 补O
                batch_tag_token = torch.nn.utils.rnn.pad_sequence(batch_tag_token, batch_first=True,
                                                                  padding_value=padding_value)
            else:
                batch_tag_token = None

            if batch_tag_token_onehot_lst:  # rz+
                if '[PAD]' in self.tag2id:
                    padding_value = self.tag2id['[PAD]']  # 补PAD 当tag2id有pad时
                else:
                    padding_value = -100  # 补O
                batch_tag_token_onehot_lst = torch.nn.utils.rnn.pad_sequence(batch_tag_token_onehot_lst,
                                                                             batch_first=True,
                                                                             padding_value=padding_value)
                tensor_list = torch.split(batch_tag_token_onehot_lst, split_size_or_sections=1, dim=0)
                batch_tag_token_onehot = [tensor.squeeze(0) for tensor in tensor_list]
                # batch_tag_token_onehot = torch.cat(batch_tag_token_onehot_lst, dim=0)
            else:
                batch_tag_token_onehot = None

            if batch_gold_label_ids:  # rz+
                if '[PAD]' in self.tag2id:
                    padding_value = self.tag2id['[PAD]']  # 补PAD 当tag2id有pad时
                else:
                    padding_value = -100  # 补O
                batch_gold_label_ids = torch.nn.utils.rnn.pad_sequence(batch_gold_label_ids, batch_first=True,
                                                                       padding_value=padding_value)
            else:
                batch_gold_label_ids = None

            if batch_distilled_task_ent_output:
                # batch_distilled_task_ent_output = torch.nn.utils.rnn.pad_sequence(batch_distilled_task_ent_output, batch_first=True, padding_value=0.)
                batch_distilled_task_ent_output = batch_distilled_task_ent_output
            else:
                batch_distilled_task_ent_output = None

            return {
                'input_ids': tensorize(batch_input_ids),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'seq_len': tensorize(batch_seq_len),
                'batch_ner_exm': batch_ner_exm,
                'batch_wo_pad_len': tensorize(batch_wo_pad_len),
                'ori_seq_len': tensorize(batch_ori_seq_len),
                'batch_ori_2_tok': tensorize(batch_ori_2_tok),

                'batch_tag_ids': batch_tag_ids,

                'batch_tag_token': batch_tag_token,

                'batch_tag_token_onehot': batch_tag_token_onehot,
                'batch_word_mask_lst': batch_word_mask_lst,
                'batch_gold_label_ids': batch_gold_label_ids,

                'batch_input_pts': tensorize(
                    batch_input_pts) if self.args.pretrain_mode == 'feature_based' else None,

                'batch_distilled_task_ent_output': batch_distilled_task_ent_output,
            }

        return {'span': span_batcher,
                'seq': seq_batcher,
                }.get(arch, None)

    def build_dataset(self, data_source, lang='ENG', arch='span', loss_type=None, train=None):
        """构造数据集"""
        if isinstance(data_source, (str, Path)):
            exm_lst = NerExample.load_from_jsonl(data_source)
        else:
            exm_lst = data_source

        if lang == 'ENG':
            for exm in exm_lst:
                if not hasattr(exm, 'ori_2_tok') or not hasattr(exm, 'bert_tok_char_lst'):
                    exm.update_to_bert_tokenize(self.tokenizer, is_split_into_words=True)
        for i, exm in enumerate(exm_lst):
            if hasattr(exm, 'bert_tok_char_lst'):
                if len(exm.bert_tok_char_lst) > self.max_len - 2:
                    print(
                        f'[index:{i}] find one exception example due to bert_tok_char_lst longer then max_len({self.max_len})')
                    exm.truncate_by_bert_tok_char_lst(max_size=self.max_len - 2, direction='tail')
                    # print(f'strip one example due to bert_tok_char_lst longer then max_len({max_len})')
                    # continue
            else:
                exm.truncate(max_size=self.max_len - 2, direction='tail')

        if loss_type is None:
            loss_type = self.loss_type
        if train:
            return LazyDataset(exm_lst, self.post_process,
                               post_process_args=dict(lang=lang, train=True, arch=arch, loss_type=loss_type)
                               )
        else:
            return LazyDataset(exm_lst, self.post_process,
                               post_process_args=dict(lang=lang, train=False, arch=arch, loss_type=loss_type)
                               )


# class LazyDataset(DataLoaderX):
class LazyDataset(torch.utils.data.Dataset):
    """LazyDataset"""

    def __init__(self, instances, post_process_fn, post_process_args):
        self.instances = instances
        self.post_process_fn = post_process_fn
        self.post_process_args = post_process_args

    def __getitem__(self, idx):
        """Get the instance with index idx"""
        return self.post_process_fn(self.instances[idx],
                                    **self.post_process_args)  # 在DataLoader的时候才对输入进行处理(wrapper) 所以叫Lazy

    def __len__(self):
        return len(self.instances)

    def __str__(self):
        return f"<LazyDataset> Num:{len(self)}"

    def __repr__(self):
        return str(self)
