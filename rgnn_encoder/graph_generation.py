import pandas as pd
import torch
import json
import numpy as np

ENTITY_NUM = 150
RELATION_NUM = 50

with open('./data/VG-KB/VG-SGG-dicts-with-attri.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    label2id = json_data['label_to_idx']
    predicate2id = json_data['predicate_to_idx']
    id2predicate = sorted(predicate2id, key=lambda k: predicate2id[k]) #-1
    id2label = sorted(label2id, key=lambda k: label2id[k])#-1
    id2predicate_word = []
    predicate_words = []
    for item in id2predicate:
        words = item.split(' ')
        predicate_words += words
        id2predicate_word.append(words)
    predicate_words = list(set(predicate_words))

def generate_entity2id():
    with open('./data/VG-KB/entity2id.txt', "w") as f:
        for i in range(ENTITY_NUM):
            write_str = id2label[i] + '\t' + str(i)
            if i < ENTITY_NUM-1:
                write_str += '\n'
            f.write(write_str)

def generate_relation2id():
    with open('./data/VG-KB/relation2id.txt', "w") as f:
        for i in range(RELATION_NUM):
            write_str = '_'.join(id2predicate_word[i]) + '\t' + str(i)
            if i < RELATION_NUM-1:
                write_str += '\n'
            f.write(write_str)



def generate_dataset(data_type):
    with open("./data/VG-KB/%s_idx.txt" % (data_type), "r") as f:
        data = f.read().strip().split("\n")
        # print(data)
    with open('./data/VG-KB/%s.txt' % (data_type), "w") as f:
        for triple_str in data:
            triple = triple_str.split("\t")
            # print(triple)
            write_str = "%s\t%s\t%s\n" % (id2label[int(triple[0])-1],'_'.join(id2predicate_word[int(triple[1])-1]),id2label[int(triple[2])-1])
            f.write(write_str)



def generate_vec():
    label_word_embedding = {}
    predicate_word_embedding = {}
    with open("./data/VG-KB/glove.6B.200d.txt", "r") as f:
        line = f.readline()
        while line:
            data = line.split(' ')
            if data[0] in id2label:
                label_word_embedding[data[0]] = [float(item) for item in data[1:]]
            if data[0] in predicate_words:
                predicate_word_embedding[data[0]] = [float(item) for item in data[1:]]
            line = f.readline()

    entity_embeddings = []
    relation_embeddings = []
    for word in id2label:
        entity_embeddings.append(label_word_embedding[word])
    for words in id2predicate_word:
        embeddings = [predicate_word_embedding[word] for word in words]
        relation_embeddings.append(np.sum(embeddings,axis=0).tolist())

    with open('./data/VG-KB/entity2vec.txt', "w") as f:
        for i,vectors in enumerate(entity_embeddings):
            write_str = ''
            for vector in vectors:
                write_str += str(vector)
                write_str += '\t'
            if i < ENTITY_NUM-1:
                write_str += '\n'
            f.write(write_str)

    with open('./data/VG-KB/relation2vec.txt', "w") as f:
        for i,vectors in enumerate(relation_embeddings):
            write_str = ''
            for vector in vectors:
                write_str += str(vector)
                write_str += '\t'
            if i < RELATION_NUM-1:
                write_str += '\n'
            f.write(write_str)


def generate_we():
    label_word_embedding = {}
    with open("./data/VG-KB/glove.6B.200d.txt", "r") as f:
        line = f.readline()
        while line:
            data = line.split(' ')
            if data[0] in id2label:
                label_word_embedding[data[0]] = [float(item) for item in data[1:]]
            line = f.readline()

    entity_embeddings = []
    for word in id2label:
        entity_embeddings.append(label_word_embedding[word])

    with open('./data/VG-KB/we/glove.6B.200d.txt', "w") as f:
        for i,vectors in enumerate(entity_embeddings):
            # print(np.linalg.norm(np.array(vectors)))
            write_str = id2label[i] + ' ' + ' '.join(np.array(vectors).astype(str).tolist())
            if i < len(entity_embeddings)-1:
                write_str += '\n'
            f.write(write_str)


# generate_entity2id()
# generate_relation2id()
# generate_dataset('train')
# generate_dataset('valid')
# generate_dataset('test')
# generate_vec()
# generate_we()
