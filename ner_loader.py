import sys, random, os
from typing import *
# import ipdb
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import numpy as np
import copy
import datautils as utils
import json
from datautils import NerExample
from data_reader import NerDataReader


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements `sequentially` from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


# curr_dir = os.path.dirname(__file__)
# parent_dir = os.path.dirname(curr_dir)
data_dir = 'data/'


def analyse_task_ent_dist():
    exm_file = data_dir + 'onto/train_task.jsonl'
    exm_lst = NerExample.load_from_jsonl(exm_file, token_deli=' ',
                                         external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
    task_exms_dct = {}
    for exm in exm_lst:
        task_id = str(exm.task_id)
        task_exms_dct.setdefault(task_id, []).append(exm)
    for task_id in sorted(task_exms_dct):
        print(f'===task: {task_id}===')
        NerExample.stats(task_exms_dct[task_id])


def sort_by_idx(lst, ids=None):
    if ids is None:
        return lst
    sorted_lst = []
    for idx in ids:
        sorted_lst.append(lst[idx])
    return sorted_lst


class CL_Loader:
    def __init__(self, args,
                 entity_task_lst,
                 ori_label_token_map,   # rz+
                 split_seed=1,
                 max_len=512,
                 # bert_model_dir='huggingface_model_resource/bert-base-cased'):
                 ):
        self.bert_model_dir = args.bert_model_dir
        self.split_seed = args.split_seed
        self.max_len = max_len
        self.entity_task_lst = entity_task_lst  # 最好在这里的时候就变为具有perm信息的结果
        self.ori_label_token_map = ori_label_token_map  # rz+
        self.num_tasks = len(self.entity_task_lst)
        self.tid2ents = {tid: ents for tid, ents in enumerate(self.entity_task_lst)}
        self.num_ents_per_task = [len(ents) for ents in self.entity_task_lst]
        self.ent2tid = {}
        for tid, ents in self.tid2ents.items():
            for ent in ents:
                self.ent2tid[ent] = tid

        self.entity_lst = sum(self.entity_task_lst, [])
        self.datareader = NerDataReader(self.bert_model_dir, self.max_len, self.ori_label_token_map, ent_file_or_ent_lst=self.entity_lst)

        self.ent2id = self.datareader.ent2id
        self.tid2entids = {tid: [self.ent2id[ent] for ent in ents] for tid, ents in self.tid2ents.items()}
        self.tid2offset = {tid: [min(entids), max(entids) + 1] for tid, entids in self.tid2entids.items()}
        self.args = args   # rz+
        print('tid2offset', self.tid2offset)
        print('id2ent', self.datareader.id2ent)
        print('tid2entids', self.tid2entids)

    def init_data(self, datafiles=None, setup=None, bsz=14, test_bsz=64, arch='span', use_pt=False, gpu=True, quick_test=False):
        self.task_train_generator = torch.Generator()  # make sure no affect by model_init (teacher model) i.e. non_cl_task0 = cl_task0
        # to make sure it's the same e.g. train single task 6 above task 5 = train task 1-6  # 要先蒸馏消耗g 再训练也消耗g
        # self.task_train_generators = [torch.Generator() for _ in range(self.num_tasks)]
        # [g.manual_seed(i) for i, g in enumerate(self.task_train_generators)]
        # 当时出现的问题是 每轮测试还是最后一轮测试竟然影响模型的随机性，因为只要dataloader被迭代一次都会消耗一次全局种子的随机次数。

        # setup: split or filter
        self.bsz = bsz
        self.test_bsz = test_bsz
        self.arch = arch
        self.gpu = gpu
        if setup is None:
            setup = self.setup
        else:
            self.setup = setup
        if datafiles is None:
            datafiles = self.datafiles

        if not quick_test:
            """train"""
            self.train_exm_lst, self.train_tid2exmids = self.load_data_with_taskid(data_dir + datafiles['train'], data_dir + datafiles['dev'], data_dir + datafiles['test'],
                                                                                   setup=setup, use_pt=use_pt, fewshot=False, CL_fewshot=True, args=self.args)

        """dev"""  # 少样本CL设置中验证集不用
        self.dev_exm_lst, self.dev_tid2exmids = self.load_data_with_taskid(data_dir + datafiles['train'], data_dir + datafiles['dev'], data_dir + datafiles['test'],
                                                                           setup=setup, use_pt=use_pt, args=self.args)
        if quick_test:
            self.train_exm_lst = self.dev_exm_lst
            self.train_tid2exmids = self.dev_tid2exmids

        """test"""  # for Test Filter
        self.test_exm_lst, self.test_tid2exmids = self.load_data_with_taskid(data_dir + datafiles['train'], data_dir + datafiles['dev'], data_dir + datafiles['test'],
                                                                             setup='filter', use_pt=use_pt, args=self.args)

        self.num_train = len(self.train_exm_lst)
        self.num_dev = len(self.dev_exm_lst)
        self.num_test = len(self.test_exm_lst)

        self.train_dataset = self.datareader.build_dataset(self.train_exm_lst, arch=self.arch, loss_type='sigmoid', train=True)
        self.dev_dataset = self.datareader.build_dataset(self.dev_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.test_dataset = self.datareader.build_dataset(self.test_exm_lst, arch=self.arch, loss_type='sigmoid')
        # fewnerd self.test_dateset - 1 because 1 of test_exm_lst have max_len>510
        self.init_dataloaders()

    def init_dataloaders(self):
        """ initialize dataloaders for CL"""
        setup = self.setup
        gpu = self.gpu
        self.train_tasks_dataloaders = []  # CL Train Split or Filter 这是一个总的训练集dataloader列表，包含6个，每个代表一个任务的dataloader，如：第一个表示ORG任务的，第二个表示PERSON任务的等等
        for tid in range(self.num_tasks):
            exmids = sorted(self.train_tid2exmids[tid])
            # print(f'task_id {tid} have {len(exmids)} train examples')
            dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=32 if tid == 0 else self.bsz,  # 32
                                                     sampler=SubsetRandomSampler(exmids,   # 在原始的59924条样本按照exmids中的9987个随机的索引采样
                                                                                 generator=self.task_train_generator
                                                                                 ),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),   # 这里的get_batcher_fn()方法在data_reader.py中
                                                     )
            dataset_list = dataloader.dataset.instances
            tid_indices = dataloader.sampler.indices
            tid_data = [dataset_list[idx] for idx in tid_indices]
            task_label_count = 0
            for batch in tid_data:
                ent_dct = batch.ent_dct
                # labels = batch.tag_lst
                crr_labels = self.entity_task_lst[tid]
                # 遍历当前任务的所有标签
                for crr_label in crr_labels:
                    if crr_label in ent_dct:
                        task_label_count += len(ent_dct[crr_label])

            print(f'task_id {tid} have {len(exmids)} train examples, 其中包含当前任务标签{crr_labels}的个数为: {task_label_count}')

            self.train_tasks_dataloaders.append(dataloader)

        self.dev_tasks_dataloaders = []  # CL Dev Split or Filter  # 同训练集一样
        for tid in range(self.num_tasks):
            exmids = sorted(self.dev_tid2exmids[tid])
            print(f'task_id {tid} have {len(exmids)} dev examples')
            dataloader = torch.utils.data.DataLoader(self.dev_dataset,
                                                     batch_size=self.test_bsz,  # 64
                                                     sampler=SubsetSequentialSampler(exmids),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.dev_tasks_dataloaders.append(dataloader)  # CL

        self.test_tasks_dataloaders_filtered = []  # Test Filter  # 注意！总的测试集dataloader和训练/验证是不一样的，它也是6个，但是，第一个表示ORG任务的dataloader，第二个表示ORG+PERSON的，第三个表示ORG+PERSON+GPE的，以此类推
        for tid in range(self.num_tasks):
            so_far_exmids = set()
            for i in range(0, tid + 1):
                so_far_exmids.update(self.test_tid2exmids[i])
            so_far_exmids = sorted(so_far_exmids)
            print(f'task_id {tid} have {len(so_far_exmids)} test filtered examples')
            dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                     batch_size=self.test_bsz,
                                                     sampler=SubsetSequentialSampler(so_far_exmids),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.test_tasks_dataloaders_filtered.append(dataloader)

        print(f'total {len(self.test_dataset)} test examples:')
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,  # Test All  总的测试dataloader（不按照任务划分）
                                                           batch_size=self.test_bsz,
                                                           shuffle=False,
                                                           collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                           )

        # experimental all tasks  少样本CL设置中没用到
        self.train_alltask_dataloader = torch.utils.data.DataLoader(self.train_dataset,   # 常规（不按照任务划分）的训练dataloader，包含6个任务（即6个类体类别）
                                                                    batch_size=self.bsz,
                                                                    shuffle=True,
                                                                    collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                                    # generator=self.task_train_generators[0]
                                                                    )
        # experimental all tasks  少样本CL设置中没用到
        self.dev_alltask_dataloader = torch.utils.data.DataLoader(self.dev_dataset,
                                                                  batch_size=self.test_bsz,
                                                                  shuffle=False,
                                                                  collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                                  )

        # non_CL Train
        self.so_far_train_tasks_dataloaders = [self.train_tasks_dataloaders[0]]  # need the first NonCL one align with CL first one 要第一个相当于与CL的对齐
        for tid in range(1, self.num_tasks):       # self.so_far_train_tasks_dataloaders是一个对训练集做了和CL测试集一样的操作，它也是6个，第一个表示ORG任务的dataloader，第二个表示ORG+PERSON的，第三个表示ORG+PERSON+GPE的，以此类推
            so_far_exmids = set()
            for i in range(0, tid + 1):
                so_far_exmids.update(self.train_tid2exmids[i])
            so_far_exmids = sorted(so_far_exmids)
            print(f'non_cl task_id {tid} have {len(so_far_exmids)} train examples')
            dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.bsz,
                                                     sampler=SubsetRandomSampler(so_far_exmids,
                                                                                 generator=self.task_train_generator
                                                                                 ),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.so_far_train_tasks_dataloaders.append(dataloader)
        # non_CL Dev
        self.so_far_dev_tasks_dataloaders = [self.dev_tasks_dataloaders[0]]  # 同上训练集一样
        for tid in range(1, self.num_tasks):
            so_far_exmids = set()
            for i in range(0, tid + 1):
                so_far_exmids.update(self.dev_tid2exmids[i])
            so_far_exmids = sorted(so_far_exmids)
            print(f'non_cl task_id {tid} have {len(so_far_exmids)} dev examples')
            dataloader = torch.utils.data.DataLoader(self.dev_dataset,
                                                     batch_size=self.bsz,
                                                     sampler=SubsetSequentialSampler(so_far_exmids),
                                                     collate_fn=self.datareader.get_batcher_fn(gpu=gpu, arch=self.arch),
                                                     )
            self.so_far_dev_tasks_dataloaders.append(dataloader)
        print(f'initialize CL dataloaders success, setup: {setup}')

    def reset_entity_task_lst(self, entity_task_lst):
        # when order in entity lst is permuted, we should recompute some mapping and offset for the model
        self.entity_task_lst = entity_task_lst
        self.num_tasks = len(self.entity_task_lst)
        self.tid2ents = {tid: ents for tid, ents in enumerate(self.entity_task_lst)}
        self.num_ents_per_task = [len(ents) for ents in self.entity_task_lst]
        self.ent2tid = {}
        for tid, ents in self.tid2ents.items():
            for ent in ents:
                self.ent2tid[ent] = tid

        self.entity_lst = sum(self.entity_task_lst, [])
        self.datareader = NerDataReader(self.bert_model_dir, self.max_len, ori_label_token_map=self.ori_label_token_map, ent_file_or_ent_lst=self.entity_lst)
        self.train_dataset = self.datareader.build_dataset(self.train_exm_lst, arch=self.arch, loss_type='sigmoid', train=True)  # build_dataset方法在data_reader.py中，得到的是一个LazyDataset的实例化对象
        self.dev_dataset = self.datareader.build_dataset(self.dev_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.test_dataset = self.datareader.build_dataset(self.test_exm_lst, arch=self.arch, loss_type='sigmoid')
        self.ent2id = self.datareader.ent2id
        self.tid2entids = {tid: [self.ent2id[ent] for ent in ents] for tid, ents in self.tid2ents.items()}
        self.tid2offset = {tid: [min(entids), max(entids) + 1] for tid, entids in self.tid2entids.items()}
        print('tid2offset', self.tid2offset)
        print('id2ent', self.datareader.id2ent)
        print('tid2entids', self.tid2entids)

    def permute_task_order(self, tasks_sorted_ids=None):
        # permute the task order
        if tasks_sorted_ids is None:
            return
        print('start to change learning order for the specific permutation.')
        # perm_entity_task_lst = sort_by_idx(self.entity_task_lst, tasks_sorted_ids)
        perm_entity_task_lst = self.entity_task_lst
        print('perm_entity_task_lst:', perm_entity_task_lst)
        self.reset_entity_task_lst(perm_entity_task_lst)
        # perm_mapping = {tasks_sorted_ids[i]: i for i in tasks_sorted_ids}  # 3->0, 0->4
        # print('perm_mapping', perm_mapping)
        # perm_tid2emxids = {perm_mapping[tid]: exmids for tid, exmids in self.train_tid2exmids.items()}
        # self.train_tid2exmids = perm_tid2emxids
        # perm_tid2emxids = {perm_mapping[tid]: exmids for tid, exmids in self.dev_tid2exmids.items()}
        # self.dev_tid2exmids = perm_tid2emxids
        # perm_tid2emxids = {perm_mapping[tid]: exmids for tid, exmids in self.test_tid2exmids.items()}
        # self.test_tid2exmids = perm_tid2emxids
        self.init_dataloaders()

    # 统计实体频率
    def count_entity_frequency(self, data, entity_lst):
        freq = {}
        for item in data:
            for ent_type in item.ent_dct:
                if ent_type in entity_lst:  # 只统计感兴趣的实体类别
                    if ent_type not in freq:
                        freq[ent_type] = 0
                    freq[ent_type] += len(item.ent_dct[ent_type])
        return freq

    # 贪婪采样
    def greedy_sampling(self, data, freq, K):
        sorted_entities = sorted(freq, key=lambda x: freq[x])  # 按频率排序
        support_set = []
        counts = {ent: 0 for ent in sorted_entities}
        random.seed(self.split_seed)
        random.shuffle(data)

        for ent in sorted_entities:
            for item in data:
                if hasattr(item, 'task_id'):  # 防止已经采样的数据再进行重采样
                    continue
                else:
                    # if len(item.char_lst) > 15:  # 尽量去除掉那种无意义的句子
                    if ent in item.ent_dct and counts[ent] < K:
                        item.remove_ent_by_type(self.entity_lst, input_keep=True)
                        item.task_id = self.entity_lst.index(ent)
                        support_set.append(item)
                        # counts[ent] += 1  # 5个句子
                        counts[ent] += len(item.ent_dct[ent])  # 5个label
                        if counts[ent] == K:
                            break
        return support_set

    # 定义实体计数函数
    def count_entity_from_ner_example(self, example):
        ent_dct = example.ent_dct
        return {ent_type: len(entities) for ent_type, entities in ent_dct.items()}

    # 定义采样函数
    def sample_ner_data_struct_shot(self, samples, count_fn, k=1, random_state=None):
        """ sample or select a subset of samples with k
            using the sampling method from https://arxiv.org/abs/2010.02405
        Args:
            samples: list
            count_fn: input a sample, return a dict of {entity_type: count}
            k: number of entity instances for each entity type
        Returns:
            indices of the selected samples
            entity count of the selected samples
        """
        # count entities
        count = {}  # total count
        samples_count = []  # count for each sample
        for sample in samples:
            sample_count = count_fn(sample)
            samples_count.append(sample_count)
            for e_type, e_count in sample_count.items():
                count[e_type] = count.get(e_type, 0) + e_count

        # sort by entity count, iterate from the infrequent entity to the frequent and sample
        entity_types = sorted(count.keys(), key=lambda k: count[k])
        selected_ids = set()
        selected_count = {k: 0 for k in entity_types}
        random.seed(random_state)
        for entity_type in entity_types:
            while selected_count[entity_type] < k:
                samples_with_e = [i for i in range(len(samples)) if
                                  entity_type in samples_count[i] and i not in selected_ids]
                sample_id = random.choice(samples_with_e)
                selected_ids.add(sample_id)
                # update selected_count
                for e_type, e_count in samples_count[sample_id].items():
                    selected_count[e_type] += e_count

        return list(selected_ids), selected_count

    def load_data_with_taskid(self, train_exm_file, dev_exm_file, test_exm_file, setup='split', split_seed=None, use_pt=False, fewshot=None, CL_fewshot=None, args=None):
        tid2exmids = {tid: set() for tid in range(self.num_tasks)}
        few = 5   # TODO
        # 测试集的处理
        if setup == 'filter':  # task contain all exm with the required entities, non negative
            exm_lst = NerExample.load_from_jsonl(test_exm_file, token_deli=' ',
                                                 external_attrs=['bert_tok_char_lst', 'ori_2_tok'])
            for exmid, exm in enumerate(exm_lst):
                exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                for ent in exm.ent_dct:
                    tid2exmids[self.ent2tid[ent]].add(exmid)
        # 非测试集的处理
        elif setup == 'split':  # task contain a set of exm and only contain the required entities, have negative

            if split_seed is None:
                split_seed = self.split_seed
            # Compare to train.jsonl, train_task.jsonl contain task_id as attr per exm by split
            # 训练集的处理
            if fewshot:
                task_emx_file = train_exm_file.replace('.jsonl', f'_task_{few}shot_ablation_{args.perm}.jsonl')
                if os.path.exists(task_emx_file):
                    exm_lst = NerExample.load_from_jsonl(task_emx_file, token_deli=' ',
                                                         external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
                else:
                    exm_lst = NerExample.load_from_jsonl(train_exm_file, token_deli=' ',
                                                         external_attrs=['bert_tok_char_lst', 'ori_2_tok'])

                    num_data = len(exm_lst)
                    data_order = list(range(num_data))
                    random.seed(split_seed)
                    random.shuffle(data_order)

                    num_per_task = num_data // self.num_tasks
                    selected_exm_ids = set()
                    # 分好每个数据属于哪个task 然后仅保留这多个task的实体。具体每个task对应的实体在后面会自动mask
                    # allocate the data into its predefined task
                    fewshot_exm_lst = []
                    for task_id in range(self.num_tasks):
                        exm_ids_lst = []
                        # if task_id == self.num_tasks - 1:  # in case exist the remain data 除不尽的放入最后一个task
                        #     exm_ids_per_task = data_order[task_id * num_per_task:]
                        # else:
                        for exm_id, exm in enumerate(exm_lst):
                            for entity_type in self.tid2ents[task_id]:  # TODO 要兼容fewnerd
                                # if entity_type in exm.ent_dct:
                                if entity_type in exm.ent_dct and exm_id not in selected_exm_ids:       # 不要求每个任务只包含当前实体
                                # if list(exm.ent_dct.keys()) == [entity_type] and exm_id not in selected_exm_ids:        # 每个任务只包含当前实体
                                    exm_ids_lst.append(exm_id)

                        # exm_ids_per_task = data_order[task_id * num_per_task: task_id * num_per_task + num_per_task]
                        random.seed(split_seed)
                        # 确保随机选取的数量不超过列表长度
                        # num_samples = min(10, len(exm_ids_lst))
                        exm_ids_per_task = random.sample(exm_ids_lst, few)   # TODO 不要写死

                        # 将新选择的例子ID添加到selected_exm_ids中
                        selected_exm_ids.update(exm_ids_per_task)

                        for exm_id in exm_ids_per_task:
                            exm = exm_lst[exm_id]
                            # only need to keep entities used in all tasks, for ents in each task it will automaticly mask afterward (model calc_loss)
                            exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                            exm.task_id = task_id
                            fewshot_exm_lst.append(exm)
                    NerExample.save_to_jsonl(fewshot_exm_lst, task_emx_file,
                                             external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])

            elif CL_fewshot:
                task_emx_file = train_exm_file.replace('.jsonl', f'_task_CL_{few}shot_{args.perm}_{args.std}_{self.split_seed}.jsonl')
                if os.path.exists(task_emx_file):
                    exm_lst = NerExample.load_from_jsonl(task_emx_file, token_deli=' ',
                                                         external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
                else:
                    train_exm_lst = NerExample.load_from_jsonl(train_exm_file, token_deli=' ',
                                                         external_attrs=['bert_tok_char_lst', 'ori_2_tok'])
                    # train_exm_lst = random.sample(train_exm_lst, 1000)  # 减少数据量，便于调试
                    dev_exm_lst = NerExample.load_from_jsonl(dev_exm_file, token_deli=' ',
                                                         external_attrs=['bert_tok_char_lst', 'ori_2_tok'])
                    fewshot_exm_lst = []
                    # # rz+ 下面是加task0的代码：但这个程序不对，这么写是从task0的训练集中随机挑选50条数据，这些数据可能包含其余task的实体，并没有对这些进行过滤，因此结果很高，相当于就不是5shot了，而是多于5shot的样本
                    # # ================================
                    # for exm in train_exm_lst:
                    #
                    #     new_exm = copy.deepcopy(exm)  # 创建新的实例
                    #     new_exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                    #     new_exm.task_id = 0
                    #     fewshot_exm_lst.append(new_exm)
                    #
                    #     # # exm = exm_lst[exm_id]
                    #     # # only need to keep entities used in all tasks, for ents in each task it will automaticly mask afterward (model calc_loss)
                    #     # exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                    #     # exm.task_id = 0
                    #     # fewshot_exm_lst.append(exm)
                    #     # # else:
                    # freq = self.count_entity_frequency(dev_exm_lst, self.entity_lst[1:])
                    # support_set = self.greedy_sampling(dev_exm_lst, freq, K=few)
                    #
                    # # 随机抽取task0中的O类样本作为其余task的负样本
                    # for ent, task_id in self.ent2id.items():
                    #     sampled_exms = random.sample(train_exm_lst, 50)
                    #     for exm in sampled_exms:
                    #         new_exm = copy.deepcopy(exm)  # 创建新的实例
                    #         new_exm.remove_ent_by_type([ent], input_keep=True)
                    #         new_exm.task_id = task_id
                    #         support_set.append(new_exm)
                    # # rz +
                    # # ================================

                    # 下面是不加task0的代码：
                    # ================================
                    for exm in train_exm_lst:
                        # exm = exm_lst[exm_id]
                        # only need to keep entities used in all tasks, for ents in each task it will automaticly mask afterward (model calc_loss)
                        exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                        if args.std != 'filter' or any(entity in exm.ent_dct for entity in self.entity_task_lst[0]):
                            exm.task_id = 0
                            fewshot_exm_lst.append(exm)
                        # else:

                    # # # 用于存储所有任务的数据(这里是github别人复现的贪婪采样)
                    support_set = []

                    # 遍历所有task，跳过task0
                    for tid, entity_lst in self.tid2ents.items():
                        if tid == 0:
                            continue  # 跳过task0

                        curr_dev_exm_lst = []
                        for item in dev_exm_lst:
                            # if len(item.char_lst) > 20:
                            new_exm = copy.deepcopy(item)
                            # new_exm.remove_ent_by_type(entity_lst, input_keep=True)
                            curr_dev_exm_lst.append(new_exm)

                        # 进行贪婪采样，只针对当前任务所需的实体类型
                        selected_ids, selected_count = self.sample_ner_data_struct_shot(
                            curr_dev_exm_lst,
                            count_fn=self.count_entity_from_ner_example,
                            k=few,
                            random_state=self.split_seed
                        )

                        filtered_samples = [curr_dev_exm_lst[idx] for idx in selected_ids]

                        # 为每个样本添加 task_id，并将结果添加到 all_task_samples 列表中
                        for sample in filtered_samples:
                            if any(entity in sample.ent_dct for entity in entity_lst):  # 保证该任务采样的每条数据都包含当前实体
                                new_sample = copy.deepcopy(sample)
                                new_sample.task_id = tid  # 添加 task_id 属性 TODO 是否存在覆盖之前tid的情况？
                                support_set.append(new_sample)

                    # 下面这两行是我实现的贪婪采样
                    # freq = self.count_entity_frequency(dev_exm_lst, self.entity_lst[1:])
                    # support_set = self.greedy_sampling(dev_exm_lst, freq, K=few)
                    # ================================
                    fewshot_exm_lst.extend(support_set)
                    NerExample.save_to_jsonl(fewshot_exm_lst, task_emx_file,
                                             external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
                    exm_lst = fewshot_exm_lst
            # 验证集的处理
            else:
                task_emx_file = dev_exm_file.replace('.jsonl', '_task.jsonl')
                if os.path.exists(task_emx_file):
                    exm_lst = NerExample.load_from_jsonl(task_emx_file, token_deli=' ',
                                                         external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])
                else:
                    exm_lst = NerExample.load_from_jsonl(dev_exm_file, token_deli=' ',
                                                         external_attrs=['bert_tok_char_lst', 'ori_2_tok'])

                    num_data = len(exm_lst)
                    data_order = list(range(num_data))
                    random.seed(split_seed)
                    random.shuffle(data_order)

                    num_per_task = num_data // self.num_tasks

                    # 分好每个数据属于哪个task 然后仅保留这多个task的实体。具体每个task对应的实体在后面会自动mask
                    # allocate the data into its predefined task
                    for task_id in range(self.num_tasks):
                        if task_id == self.num_tasks - 1:  # in case exist the remain data 除不尽的放入最后一个task
                            exm_ids_per_task = data_order[task_id * num_per_task:]
                        else:
                            exm_ids_per_task = data_order[task_id * num_per_task: task_id * num_per_task + num_per_task]

                        for exm_id in exm_ids_per_task:
                            exm = exm_lst[exm_id]
                            # only need to keep entities used in all tasks, for ents in each task it will automaticly mask afterward (model calc_loss)
                            exm.remove_ent_by_type(self.entity_lst, input_keep=True)
                            exm.task_id = task_id
                    NerExample.save_to_jsonl(exm_lst, task_emx_file,
                                             external_attrs=['task_id', 'bert_tok_char_lst', 'ori_2_tok'])

            for exmid, exm in enumerate(exm_lst):
                if hasattr(exm, 'task_id'):        # rz+
                    tid2exmids[exm.task_id].add(exmid)

        else:
            raise NotImplementedError

        if use_pt:  # 没触发该条件，所以先不改这里
            print(f'{exm_file}.pt loading...')
            pt_lst = torch.load(f'{exm_file}.pt')
            print(f'{exm_file}.pt loaded!')
            assert len(exm_lst) == len(pt_lst)
            for exm, pt in zip(exm_lst, pt_lst):
                exm.pt = pt

        return exm_lst, tid2exmids

    def get_task_dataloader(self, mode='test', tid=None, ent=None):
        if tid is None and ent is None:
            return {
                'test': self.test_dataloader,
                'train': self.train_alltask_dataloader,
                'dev': self.dev_alltask_dataloader,
            }[mode]
        if tid is None:
            assert ent is not None
            tid = self.ent2tid[ent]

        if mode == 'train':
            return self.train_tasks_dataloaders[tid]
        if mode == 'dev':
            return self.dev_tasks_dataloaders[tid]

        raise NotImplementedError


onto_entity_task_lst = [
    ['ORG'],
    ['PERSON'],
    ['GPE'],
    ['DATE'],
    ['CARDINAL'],
    ['NORP'],
]
# onto_entity_task_lst = [
#     ['CARDINAL', 'DATE', 'EVENT', 'FAC'],
#     ['GPE', 'LANGUAGE'],
#     ['LAW'],
#     ['LOC', 'MONEY'],
#     ['NORP'],
#     ['ORDINAL', 'ORG'],
#     ['PERCENT'],
#     ['PERSON', 'PRODUCT'],
#     ['QUANTITY', 'TIME', 'WORK_OF_ART'],
# ]


# onto_ori_label_token_map = {"I-ORG": ["National", "Corp", "News", "Inc", "Senate", "Court"],
#                             "I-PERSON": ["John", "David", "Peter", "Michael", "Robert", "James"],
#                             "I-GPE": ["US", "China", "United", "Beijing", "Israel", "Taiwan"],
#                             "I-DATE": ["Year", "December", "August", "July", "1940", "March"],
#                             "I-CARDINAL": ["Two", "four", "Three", "Hundred", "20", "8"],
#                             "I-NORP": ["Chinese", "Israeli", "Palestinians", "American", "Japanese", "Palestinian"]
#                             }


class Onto_Loader(CL_Loader):
    # def __init__(self, setup, bert_model_dir='huggingface_model_resource/bert-base-cased'):
    def __init__(self, args):
        super(Onto_Loader, self).__init__(
            args=args,
            entity_task_lst=[onto_entity_task_lst[i] for i in args.perm_ids],
            ori_label_token_map=json.load(open(args.label_map_path, 'r'))
        )
        self.datafiles = {
            'train': 'onto/train.jsonl',
            'dev': 'onto/dev.jsonl',
            'test': 'onto/test.jsonl',
        }
        # self.setup = 'split'
        # self.setup = 'filter'
        self.setup = args.setup


class Onto_fs_Loader(CL_Loader):
    # def __init__(self, setup, bert_model_dir='huggingface_model_resource/bert-base-cased'):
    def __init__(self, args):
        super(Onto_fs_Loader, self).__init__(
            args=args,
            entity_task_lst=[onto_entity_task_lst[i] for i in args.perm_ids],
            ori_label_token_map=json.load(open(args.label_map_path, 'r'))
        )
        self.datafiles = {
            'train': 'onto_fs/50shot.jsonl',
            'dev': 'onto_fs/dev.jsonl',
            'test': 'onto_fs/test.jsonl',
        }
        # self.setup = 'split'
        # self.setup = 'filter'
        self.setup = args.setup


fewnerd_entity_task_lst = [
    ['location-GPE', 'location-bodiesofwater', 'location-island', 'location-mountain', 'location-other', 'location-park', 'location-road/railway/highway/transit'],
    ['person-actor', 'person-artist/author', 'person-athlete', 'person-director', 'person-other', 'person-politician', 'person-scholar', 'person-soldier'],
    ['organization-company', 'organization-education', 'organization-government/governmentagency', 'organization-media/newspaper', 'organization-other', 'organization-politicalparty', 'organization-religion', 'organization-showorganization', 'organization-sportsleague',
     'organization-sportsteam'],
    ['other-astronomything', 'other-award', 'other-biologything', 'other-chemicalthing', 'other-currency', 'other-disease', 'other-educationaldegree', 'other-god', 'other-language', 'other-law', 'other-livingthing', 'other-medical'],
    ['product-airplane', 'product-car', 'product-food', 'product-game', 'product-other', 'product-ship', 'product-software', 'product-train', 'product-weapon'],
    ['building-airport', 'building-hospital', 'building-hotel', 'building-library', 'building-other', 'building-restaurant', 'building-sportsfacility', 'building-theater'],
    ['art-broadcastprogram', 'art-film', 'art-music', 'art-other', 'art-painting', 'art-writtenart'],
    ['event-attack/battle/war/militaryconflict', 'event-disaster', 'event-election', 'event-other', 'event-protest', 'event-sportsevent'],
]


# fewnerd_ori_label_token_map = {"I-location-GPE": ["National", "Corp", "News", "Inc", "Senate", "Court"],
#                             "I-person-actor": ["John", "David", "Peter", "Michael", "Robert", "James"],
#                             "I-organization-company": ["US", "China", "United", "Beijing", "Israel", "Taiwan"],
#                             "I-other-astronomything": ["Year", "December", "August", "July", "1940", "March"],
#                             "I-product-airplane": ["Two", "four", "Three", "Hundred", "20", "8"],
#                             "I-building-airport": ["Chinese", "Israeli", "Palestinians", "American", "Japanese", "Palestinian"],
#                             "I-art-broadcastprogram": ["Chinese", "Israeli", "Palestinians", "American", "Japanese", "Palestinian"],
#                             "I-event-disaster": ["Chinese", "Israeli", "Palestinians", "American", "Japanese", "Palestinian"],
#                             }   # TODO 修改


class FewNERD_Loader(CL_Loader):
    def __init__(self, args):
        super(FewNERD_Loader, self).__init__(
            args=args,
            entity_task_lst=[fewnerd_entity_task_lst[i] for i in args.perm_ids],
            ori_label_token_map=json.load(open(args.label_map_path, 'r'))
        )
        self.datafiles = {
            'train': 'fewnerd/supervised/train.jsonl',
            'dev': 'fewnerd/supervised/dev.jsonl',
            'test': 'fewnerd/supervised/test.jsonl',
        }
        # self.setup = 'split'
        # self.setup = 'filter'
        self.setup = args.setup

conll_entity_task_lst = [
    ['PER'],
    ['LOC'],
    ['ORG'],
    ['MISC'],
]

# conll_ori_label_token_map = {"I-PER": ["Michael", "John", "David", "Thomas", "Martin", "Paul"],
#                              "I-ORG": ["Corp", "Inc", "Commission", "Union", "Bank", "Party"],
#                              "I-LOC": ["England", "Germany", "Australia", "France", "Russia", "Italy"],
#                              "I-MISC": ["Palestinians", "Russian", "Chinese", "Dutch", "Russians", "English"]
#                              }


class Conll_Loader(CL_Loader):
    # def __init__(self, setup, bert_model_dir='huggingface_model_resource/bert-base-cased'):
    def __init__(self, args):
        super(Conll_Loader, self).__init__(
            args=args,
            entity_task_lst=[conll_entity_task_lst[i] for i in args.perm_ids],  # 这种设置既适合每个任务仅包含1种实体，也适合每个任务包含多个
            ori_label_token_map=json.load(open(args.label_map_path, 'r'))
        )
        self.datafiles = {
            'train': 'conll/train.jsonl',
            'dev': 'conll/dev.jsonl',
            'test': 'conll/test.jsonl',
        }
        # self.setup = 'split'
        # self.setup = 'filter'
        self.setup = args.setup


if __name__ == '__main__':
    # analyse_task_ent_dist()
    # exit(0)

    # import datautils as utils
    #
    # utils.setup_seed(0, np, torch)
    # loader = Onto_Loader(setup='split')
    # loader.init_data(bsz=32, arch='span')
    #
    # train_dataloader1 = loader.get_task_dataloader(mode='train', tid=0)
    # dev_dataloader1 = loader.get_task_dataloader(mode='dev', tid=0)
    # for i, inputs_dct in enumerate(train_dataloader1):  # iter step
    #     print(inputs_dct['batch_ner_exm'][0])
    #     x = input()
    #     if x == 'break':
    #         break
    #
    # train_dataloader2 = loader.so_far_train_tasks_dataloaders[0]
    # dev_dataloader2 = loader.so_far_dev_tasks_dataloaders[0]
    # for i, inputs_dct in enumerate(train_dataloader2):  # iter step
    #     print(inputs_dct['batch_ner_exm'][0])
    #     x = input()
    #     if x == 'break':
    #         break

    exit(0)
