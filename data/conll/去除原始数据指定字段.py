import json

def remove_fields_from_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            # 删除指定的字段
            if "bert_tok_char_lst" in data:
                del data["bert_tok_char_lst"]
            if "ori_2_tok" in data:
                del data["ori_2_tok"]
            # 将修改后的数据写回文件
            outfile.write(json.dumps(data) + '\n')

# 使用函数处理train.jsonl文件
remove_fields_from_jsonl('test.jsonl', 'test_modified.jsonl')
