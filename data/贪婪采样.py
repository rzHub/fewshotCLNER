import random
import json

def sample_ner_data_struct_shot(samples, count_fn, k=1, random_state=None):
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
    count = {} # total count
    samples_count = [] # count for each sample
    for sample in samples:
        sample_count = count_fn(sample)
        samples_count.append(sample_count)
        for e_type, e_count in sample_count.items():
            count[e_type] = count.get(e_type, 0) + e_count

    # sort by entity count, iterate from the infrequent entity to the frequent and sample
    entity_types = sorted(count.keys(), key=lambda k: count[k])
    selected_ids = set()
    selected_count = {k:0 for k in entity_types}
    random.seed(random_state)
    for entity_type in entity_types:
        while selected_count[entity_type] < k:
            samples_with_e = [i for i in range(len(samples)) if entity_type in samples_count[i] and i not in selected_ids]
            sample_id = random.choice(samples_with_e)
            selected_ids.add(sample_id)
            # update selected_count
            for e_type, e_count in samples_count[sample_id].items():
                selected_count[e_type] += e_count

    return list(selected_ids), selected_count



from collections import Counter

def count_entity_(sample):
    return Counter([slot['label'] for slot in sample['slots']])



# 加载JSONL文件
def load_jsonl(file_path):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

# 定义实体计数函数
def count_entity_from_jsonl(sample):
    ent_dct = sample["ent_dct"]
    return {ent_type: len(entities) for ent_type, entities in ent_dct.items()}

# 加载数据集
file_path = './conll/dev.jsonl'  # 替换为你的文件路径
samples = load_jsonl(file_path)

# 进行采样
selected_ids, selected_count = sample_ner_data_struct_shot(samples, count_entity_from_jsonl, k=5, random_state=42)

# 查看结果
print("Selected sample IDs:", selected_ids)
print("Entity count in selected samples:", selected_count)

# 打印选中的样本
for idx in selected_ids:
    print(samples[idx])
