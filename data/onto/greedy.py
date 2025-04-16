import json


# 读取数据，指定文件编码为 UTF-8
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [json.loads(line) for line in lines]
    return data


# 统计实体频率
def count_entity_frequency(data, entity_lst):
    freq = {}
    for item in data:
        for ent_type in item['ent_dct']:
            if ent_type in entity_lst:  # 只统计感兴趣的实体类别
                if ent_type not in freq:
                    freq[ent_type] = 0
                freq[ent_type] += len(item['ent_dct'][ent_type])
    return freq


# 贪婪采样
def greedy_sampling(data, freq, K):
    sorted_entities = sorted(freq, key=lambda x: freq[x])  # 按频率排序
    support_set = []
    counts = {ent: 0 for ent in sorted_entities}

    for ent in sorted_entities:
        for item in data:
            if ent in item['ent_dct'] and counts[ent] < K:
                support_set.append(item)
                counts[ent] += 1
                if counts[ent] == K:
                    break
    return support_set


# 保存采样结果到文件
def save_to_file(support_set, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in support_set:
            file.write(json.dumps(item) + '\n')


# 主执行函数
def main():
    file_path = 'dev.jsonl'
    output_file_path = 'sampled_dev.jsonl'
    entity_lst = ['ORG', 'PERSON', 'GPE', 'DATE', 'CARDINAL', 'NORP']  # 感兴趣的实体列表
    data = load_data(file_path)
    freq = count_entity_frequency(data, entity_lst)
    support_set = greedy_sampling(data, freq, K=5)
    save_to_file(support_set, output_file_path)
    print(f"Sampled data has been saved to {output_file_path}")


if __name__ == '__main__':
    main()
