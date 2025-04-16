import json

def conll_to_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data_list = []
    sentence_data = {"text": "", "ent_dct": {}, "char_lst": [], "bert_tok_char_lst": [], "ori_2_tok": []}

    for line in lines:
        if line == '\n':
            if sentence_data["text"]:
                data_list.append(sentence_data)
            sentence_data = {"text": "", "ent_dct": {}, "char_lst": [], "bert_tok_char_lst": [], "ori_2_tok": []}
        else:
            tokens = line.strip().split('\t')
            word, label = tokens[0], tokens[1]

            sentence_data["text"] += word + " "
            sentence_data["char_lst"].append(word)

            if label != 'O':
                entity_type, entity_label = label.split('-')
                if entity_type not in sentence_data["ent_dct"]:
                    sentence_data["ent_dct"][entity_type] = []
                sentence_data["ent_dct"][entity_type].append([len(sentence_data["char_lst"]) - len(word), len(sentence_data["char_lst"]) - 1])

            sentence_data["ori_2_tok"].append(len(sentence_data["bert_tok_char_lst"]))

            # Tokenization using BERT tokenizer (if needed)
            # bert_tokens = bert_tokenizer.tokenize(word)
            # sentence_data["bert_tok_char_lst"].extend(bert_tokens)

    return data_list

def save_jsonl(data_list, output_path):
    with open(output_path, 'w', encoding='utf-8') as json_file:
        for data in data_list:
            json.dump(data, json_file, ensure_ascii=False)
            json_file.write('\n')

conll_file_path = 'train.txt'  # Replace with the actual path to your Conll03 file
jsonl_output_path = 'output.jsonl'  # Replace with the desired output path

data_list = conll_to_jsonl(conll_file_path)
save_jsonl(data_list, jsonl_output_path)
