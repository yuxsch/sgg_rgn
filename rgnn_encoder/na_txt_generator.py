import torch
import numpy as np
import json

ENTITY_NUM = 150
RELATION_NUM = 50
DIM = 200


def generate_entity(embeddings,emb_type):
    with open('./data/VG-KB/VG-SGG-dicts-with-attri.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        label2id = json_data['label_to_idx']
        id2label = sorted(label2id, key=lambda k: label2id[k])  # -1

    with open('./data/VG-KB/%s/glove.6B.200d.txt' % emb_type, "w") as f:
        for i in range(ENTITY_NUM):
            # print(embeddings[i].norm())
            write_str = id2label[i] + ' ' + ' '.join(embeddings[i].numpy().astype(str).tolist())
            if i < ENTITY_NUM - 1:
                write_str += '\n'
            f.write(write_str)


def generate_relation(embeddings,emb_type):
    with open('./data/VG-KB/VG-SGG-dicts-with-attri.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        predicate2id = json_data['predicate_to_idx']
        id2predicate = sorted(predicate2id, key=lambda k: predicate2id[k])  # -1
        id2predicate_word = []
        predicate_words = []
        for item in id2predicate:
            words = item.split(' ')
            predicate_words += words
            id2predicate_word.append(words)
        predicate_words = list(set(predicate_words))

    with open('./data/VG-KB/%s/relation.6B.200d.txt' % emb_type, "w") as f:
        for i in range(RELATION_NUM):
            write_str = '_'.join(id2predicate_word[i]) + ' ' + ' '.join(embeddings[i].numpy().astype(str).tolist())
            if i < RELATION_NUM - 1:
                write_str += '\n'
            f.write(write_str)


embeddings = torch.randn(ENTITY_NUM, DIM)
generate_entity(embeddings, 'random')

# embeddings = torch.randn(RELATION_NUM, DIM)
# generate_relation(embeddings, 'random')