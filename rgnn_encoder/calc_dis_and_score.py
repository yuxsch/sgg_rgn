import torch
import numpy as np
import json

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

def calc_similar(emb_type):
    embeddings = []
    with open('./data/VG-KB/%s/glove.6B.200d.txt' % emb_type, "r") as f:
        lines = f.read().split('\n')
    for line in lines:
        items = line.split(' ')
        embeddings.append(np.array(items[1:]).astype(float))


    dis = np.zeros([ENTITY_NUM,ENTITY_NUM])
    with open('./data/VG-KB/%s/dis.txt' % emb_type, "w") as f:
        for i in range(ENTITY_NUM):
            for j in range(ENTITY_NUM):
                en1 = id2label[i]
                en2 = id2label[j]
                emb1 = embeddings[i]
                emb2 = embeddings[j]
                # print(np.linalg.norm(emb2))
                # For the embeddings whose norms are equal to 1, the cosine distance is equivalent to the euler distance.
                # d = np.sum(np.abs(emb1-emb2))
                d = np.linalg.norm((emb1 - emb2))
                dis[i,j] = d
                f.write("%s-%s: %f \n"%(en1,en2,d))
    index = np.argsort(dis, axis=1)
    rank = np.argsort(index, axis=1)
    rank_entity = np.zeros([ENTITY_NUM,ENTITY_NUM]).astype(int)
    for i in range(ENTITY_NUM):
        for j in range(ENTITY_NUM):
            r = rank[i,j]
            rank_entity[i,r] = j
    with open('./data/VG-KB/%s/dis_rank.txt' % emb_type, "w") as f:
        for i in range(ENTITY_NUM):
            for r in range(ENTITY_NUM):
                en1 = id2label[i]
                j = rank_entity[i,r]
                en2 = id2label[j]
                f.write("%s-%s: %d \n"%(en1,en2,r))

    # with open('./data/VG-KB/%s/dis_rank.txt' % emb_type, "w") as f:
    #     for i in range(ENTITY_NUM):
    #         for j in range(ENTITY_NUM):
    #             en1 = id2label[i]
    #             en2 = id2label[j]
    #             r = rank[i,j]
    #             f.write("%s-%s: %d \n"%(en1,en2,r))

calc_similar('na')

def calc_score(emb_type):
    entity_embeddings = []
    relation_embeddings = []
    with open('./data/VG-KB/%s/glove.6B.200d.txt' % emb_type, "r") as f:
        entity_lines = f.read().split('\n')
    for line in entity_lines:
        items = line.split(' ')
        entity_embeddings.append(np.array(items[1:]).astype(float))
    with open('./data/VG-KB/%s/relation.6B.200d.txt' % emb_type, "r") as f:
        relation_lines = f.read().split('\n')
    for line in relation_lines:
        items = line.split(' ')
        relation_embeddings.append(np.array(items[1:]).astype(float))

    dis = np.zeros([ENTITY_NUM,ENTITY_NUM,RELATION_NUM])
    with open('./data/VG-KB/%s/score.txt' % emb_type, "w") as f:
        for i in range(ENTITY_NUM):
            print("%d:%d" % (i,ENTITY_NUM))
            for j in range(ENTITY_NUM):
                for k in range(RELATION_NUM):
                    en1 = id2label[i]
                    en2 = id2label[j]
                    predicate = '_'.join(id2predicate_word[k])

                    emb1 = entity_embeddings[i]
                    emb2 = entity_embeddings[j]
                    emb_pred = relation_embeddings[k]
                    d = np.sum(np.abs(emb1 + emb_pred - emb2))
                    dis[i,j,k] = d
                    f.write("%s-%s-%s: %f \n"%(en1,predicate,en2,d))
    index = np.argsort(dis, axis=2)
    rank = np.argsort(index, axis=2)
    rank_relation = np.zeros([ENTITY_NUM,ENTITY_NUM,RELATION_NUM]).astype(int)
    for i in range(ENTITY_NUM):
        for j in range(ENTITY_NUM):
            for k in range(RELATION_NUM):
                r = rank[i,j,k]
                rank_relation[i,j,r] = k
    with open('./data/VG-KB/%s/score_rank.txt' % emb_type, "w") as f:
        for i in range(ENTITY_NUM):
            for j in range(ENTITY_NUM):
                for r in range(RELATION_NUM):
                    en1 = id2label[i]
                    en2 = id2label[j]
                    k = rank_relation[i,j,r]
                    predicate = '_'.join(id2predicate_word[k])
                    f.write("%s-%s-%s: %d \n"%(en1,predicate,en2,r))

    # with open('./data/VG-KB/%s/score_rank.txt' % emb_type, "w") as f:
    #     for i in range(ENTITY_NUM):
    #         print("%d:%d" % (i,ENTITY_NUM))
    #         for j in range(ENTITY_NUM):
    #             for k in range(RELATION_NUM):
    #                 en1 = id2label[i]
    #                 en2 = id2label[j]
    #                 predicate = '_'.join(id2predicate_word[k])
    #                 r = rank[i,j,k]
    #                 f.write("%s-%s-%s: %d \n"%(en1,predicate,en2,r))


# calc_score('na')