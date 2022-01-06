import pandas as pd
import torch
import json



with open('./tuples/VG-SGG-dicts-with-attri.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    label2id = json_data['label_to_idx']
    predicate2id = json_data['predicate_to_idx']
    id2predicate = sorted(predicate2id, key=lambda k: predicate2id[k]) #-1
    id2label = sorted(label2id, key=lambda k: label2id[k])#-1


pzrm_imp = torch.load("./tuples/imp/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))
pzrm_tde = torch.load("./tuples/tde/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))


pzrm_motif = torch.load("./tuples/motif/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))
pzrm_motif_we = torch.load("./tuples/motif-we/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))
pzrm_motif_rgn = torch.load("./tuples/motif-rgn/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))

pzrm_vctree = torch.load("./tuples/vctree/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))
pzrm_vctree_we = torch.load("./tuples/vctree-we/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))
pzrm_vctree_rgn = torch.load("./tuples/vctree-rgn/predicate_zero_recall_mean.pytorch", map_location=torch.device("cpu"))



# print(pzrm_imp)

for i in range(50):
    s_o = "%s : imp(%.4f), tde(%.4f)" % (id2predicate[i], pzrm_imp[i], pzrm_tde[i])
    print(s_o)
    s_m = "%s : motif(%.4f), motif-we(%.4f), motif-rgn(%.4f)" % (id2predicate[i], pzrm_motif[i], pzrm_motif_we[i], pzrm_motif_rgn[i])
    print(s_m)
    s_v = "%s : vctree(%.4f), vctree-we(%.4f), vctree-rgn(%.4f)" % (id2predicate[i], pzrm_vctree[i], pzrm_vctree_we[i], pzrm_vctree_rgn[i])

    print(s_v)