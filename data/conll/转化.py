import json
from transformers import BertTokenizerFast

def process_jsonl_file(input_file, output_file):
    # 初始化分词器，关闭自动小写转换
    tokenizer = BertTokenizerFast.from_pretrained('../../PLM/bert-base-cased', do_lower_case=False)

    with open(input_file, 'r', encoding='utf-8') as file, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in file:
            data = json.loads(line)
            text = data['text']
            char_lst = data['char_lst']

            # 分词并返回偏移量映射
            encoded_input = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
            offsets = encoded_input['offset_mapping']

            # 初始化映射列表
            ori_to_tok_map = list(range(len(char_lst)))  # 直接使用原始索引作为映射

            data['bert_tok_char_lst'] = tokens
            data['ori_2_tok'] = ori_to_tok_map

            # 将更新后的字典转换为 JSON 字符串并写入输出文件
            json_line = json.dumps(data)
            outfile.write(json_line + '\n')

# 调用处理函数
process_jsonl_file('test1.jsonl', 'test.jsonl')





# import json
# from transformers import BertTokenizer
#
#
# def add_bert_tokens(entry):
#     tokenizer_path = './bert-base-cased'
#     tokenizer = BertTokenizer.from_pretrained(tokenizer_path, use_fast=True, do_lower_case=False)
#     # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     text = entry["text"]
#
#     # Tokenize input text
#     bert_tokens = tokenizer.tokenize(text)
#
#     # # Remove [CLS] and [SEP] tokens
#     # bert_tokens = [token for token in bert_tokens if token not in ['[CLS]', '[SEP]']]
#
#     # Create original to tokenized mapping
#     ori_to_tok = []
#     current_token = ""
#     for i, char in enumerate(text):
#         if char == ' ':
#             current_token += char
#         else:
#             current_token += char
#             if current_token == bert_tokens[0]:
#                 ori_to_tok.append(i)
#                 bert_tokens.pop(0)
#                 current_token = ""
#
#     entry["bert_tok_char_lst"] = bert_tokens
#     entry["ori_2_tok"] = ori_to_tok
#
#     return entry
#
#
# def process_jsonl(input_file, output_file):
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = [json.loads(line) for line in f]
#
#     processed_data = [add_bert_tokens(entry) for entry in data]
#
#     with open(output_file, 'w', encoding='utf-8') as output:
#         for entry in processed_data:
#             output.write(json.dumps(entry, ensure_ascii=False) + '\n')
#
#
# # 调用函数
# process_jsonl('test_o.jsonl', 'test.jsonl')
