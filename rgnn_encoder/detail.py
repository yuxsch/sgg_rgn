

import numpy as np
import torch
import json
from functools import reduce


N = 20

with open('./tuples/image_data.json','r',encoding='utf8')as fp:
    image_info = json.load(fp)

with open('./tuples/VG-SGG-dicts-with-attri.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    # # print(json_data.keys())
    # id2label = json_data['idx_to_label']
    label2id = json_data['label_to_idx']
    # id2predicate = json_data['idx_to_predicate']
    predicate2id = json_data['predicate_to_idx']
    id2predicate = sorted(predicate2id, key=lambda k: predicate2id[k]) #-1
    id2label = sorted(label2id, key=lambda k: label2id[k])#-1


zeroshot_triplet = torch.load("./tuples/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()
# subject,object,predicate


zeroshot_tuples = []
for item in zeroshot_triplet:
    zeroshot_tuples.append(tuple([item[0],item[2],item[1]]))
    # print(id2label[t[0]-1],id2predicate[t[2]-1],id2label[t[1]-1])
# print(zeroshot_tuple)

def list2tuple(ls):
    tuples = []
    for sub_ls in ls:
        sub_tuples = []
        for item in sub_ls:
            sub_tuples.append(tuple(item))
        tuples.append(sub_tuples)
    return tuples




# def calcRecall():
#     for i in range(len(gt_tuples)):
#         print(str(i) + ' / ' + str(len(gt_tuples)))
#         per_pred_to_gt = pred_to_gt[i]
#         gt_tris = gt_tuples[i]
#         match = reduce(np.union1d, per_pred_to_gt[:N])
#         rec_i = float(len(match)) / float(len(gt_tris))
#         recall.append(rec_i)
#     print(np.mean(recall))

def calcPredZeroRecall(model):
    gt_triples = torch.load("./tuples/" + model +  "/gt_triples.pytorch", map_location=torch.device("cpu"))
    # subject, predicate, object
    # pred_triples = torch.load("./pred_triples.pytorch", map_location=torch.device("cpu"))
    pred_to_gt = torch.load("./tuples/" + model + "/pred_to_gt.pytorch", map_location=torch.device("cpu"))

    gt_tuples = list2tuple(gt_triples)
    # pred_tuples = list2tuple(pred_triples)
    #
    # recall = []
    predicate_zero_recall = [[] for _ in range(50)]

    for i in range(len(gt_tuples)):
        print(str(i) + ' / ' + str(len(gt_tuples)))
        per_pred_to_gt = pred_to_gt[i]
        gt_tris = gt_tuples[i]
        match = reduce(np.union1d, per_pred_to_gt[:N])

        recall_hit = [0] * 50
        recall_count = [0] * 50

        for idx in range(len(gt_tris)):
            pred_label = gt_tris[idx][1]
            tri = gt_tris[idx]
            if tri in zeroshot_tuples:
                recall_count[int(pred_label)-1] += 1

        for idx in range(len(match)):
            pred_label = gt_tris[int(match[idx])][1]
            tri = gt_tris[int(match[idx])]
            if tri in zeroshot_tuples:
                recall_hit[int(pred_label)-1] += 1

        for pred_idx in range(50):
            if recall_count[pred_idx] > 0:
                predicate_zero_recall[pred_idx].append(float(recall_hit[pred_idx] / recall_count[pred_idx]))

    predicate_zero_recall_mean = []
    for pred_idx in range(50):
        m = np.mean(predicate_zero_recall[pred_idx])
        predicate_zero_recall_mean.append(m)
        print(m)
    torch.save(predicate_zero_recall_mean, "./tuples/" + model + '/predicate_zero_recall_mean.pytorch')


calcPredZeroRecall('imp')
calcPredZeroRecall('tde')
calcPredZeroRecall('motif')
calcPredZeroRecall('motif-we')
# calcPredZeroRecall('motif-rgn')
# calcPredZeroRecall('vctree')
# calcPredZeroRecall('vctree-we')
# calcPredZeroRecall('vctree-rgn')



# print(gt_tuples)
#
# hit_predicate_cnt = np.zeros(50)
# predicate_cnt = np.zeros(50)
#
# hit_triple_cnt = np.zeros(len(zeroshot_tuples))
# triple_cnt = np.zeros(len(zeroshot_tuples))
#
#
# #zero-shot
#
# for i in range(len(gt_tuples)):
#     image_file_name = str(image_info[i]['image_id']) + '.jpg'
#     print(str(i) + ' / ' + str(len(gt_tuples)))
#     # print(image_file_name)
#     gt_tris = gt_tuples[i]
#     pred_tris = pred_triples[i][:N]
#     for t in gt_tris:
#         if t in zeroshot_tuples:
#             # 此处判断所有的zero-shot三元组
#             index = zeroshot_tuples.index(t)
#             triple_cnt[index] += 1
#             predicate_cnt[t[1]-1] += 1
#
#             if t in pred_tris:
#                 # 此处判断成功判断的zero-shot三元组
#                 hit_triple_cnt[index] += 1
#                 hit_predicate_cnt[t[1] - 1] += 1
#                 print(id2label[t[0] - 1], id2predicate[t[1] - 1], id2label[t[2] - 1], '\t True')
#             else:
#                 print(id2label[t[0] - 1], id2predicate[t[1] - 1], id2label[t[2] - 1], '\t False')

# for i in range(len(zeroshot_tuples)):
#     print(str(i) + ':  ' + str(hit_triple_cnt[i] / triple_cnt[i]))
#
# for i in range(50):
#     print(str(i) + ':  ' + str(hit_predicate_cnt[i] / predicate_cnt[i]))
