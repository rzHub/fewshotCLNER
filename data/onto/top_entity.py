import json
from collections import defaultdict, Counter

def get_top_entities(filename, top_n=6):
    # 初始化一个字典用于存储每类实体及其对应的Counter
    entity_counter = defaultdict(Counter)

    # 读取并解析JSONL文件
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            char_lst = data['char_lst']
            ent_dct = data['ent_dct']

            # 遍历每类实体
            for ent_type, spans in ent_dct.items():
                for span in spans:
                    start_idx, end_idx = span
                    # 提取实体词
                    entity = " ".join(char_lst[start_idx:end_idx])
                    # 统计实体词出现的频率
                    entity_counter[ent_type][entity] += 1

    # 找出每类实体中频率最高的前top_n个实体
    top_entities = {ent_type: counter.most_common(top_n) for ent_type, counter in entity_counter.items()}

    return top_entities

# 使用示例
filename = 'train.jsonl'
top_entities = get_top_entities(filename, top_n=6)

# 输出结果
for ent_type, entities in top_entities.items():
    print(f"Top {len(entities)} entities for {ent_type}:")
    for entity, count in entities:
        print(f"{entity}: {count}")
    print()
