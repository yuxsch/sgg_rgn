import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, relation_dim, dropout, alpha, nheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions
            nhid  -> Entity Output Embedding dimensions
            relation_dim -> Relation Embedding dimensions
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed,
                edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop):
        x = entity_embeddings
        if len(edge_type_nhop) == 0:
            edge_embed_nhop = torch.tensor([],dtype=edge_type_nhop.dtype,device=edge_type_nhop.device)
        else:
            edge_embed_nhop = relation_embed[edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]

        x = torch.cat([att(x, edge_list, edge_embed, edge_list_nhop, edge_embed_nhop)
                       for att in self.attentions], dim=1)
        x = self.dropout_layer(x)

        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        if len(edge_type_nhop) == 0:
            edge_embed_nhop = torch.tensor([], dtype=edge_type_nhop.dtype, device=edge_type_nhop.device)
        else:
            edge_embed_nhop = out_relation_1[edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]

        x = F.elu(self.out_att(x, edge_list, edge_embed,
                               edge_list_nhop, edge_embed_nhop))
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, depth_gat, score_layers, not_res):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]


        self.entity_out_dim = entity_out_dim
        self.nheads_GAT = nheads_GAT
        self.depth_gat = depth_gat
        self.score_layers = score_layers
        self.score_num = len(score_layers)
        self.not_res = not_res

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim = entity_out_dim

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim * self.nheads_GAT))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim * self.nheads_GAT))

        if self.score_num > 0:
            self.record_entity_embeddings = nn.Parameter(
                torch.randn(self.num_nodes, self.score_num, self.entity_out_dim * self.nheads_GAT))

            self.record_relation_embeddings = nn.Parameter(
                torch.randn(self.num_relation, self.score_num, self.entity_out_dim * self.nheads_GAT))



        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gats = nn.ModuleList()
        for i in range(depth_gat):
            if i == 0:
                gat = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim, self.relation_dim, self.drop_GAT, self.alpha, self.nheads_GAT)
            else:
                gat = SpGAT(self.num_nodes, self.entity_out_dim * self.nheads_GAT, self.entity_out_dim * self.nheads_GAT, self.relation_out_dim * self.nheads_GAT, self.drop_GAT, self.alpha, 1)
            self.sparse_gats.append(gat)
        # self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim, self.relation_dim,
        #                           self.drop_GAT, self.alpha, self.nheads_GAT)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim,  self.entity_out_dim * self.nheads_GAT)))
        self.W_relations = nn.Parameter(torch.zeros(
            size=(self.relation_dim,  self.relation_out_dim * self.nheads_GAT)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)
        nn.init.xavier_uniform_(self.W_relations.data, gain=1.414)

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop):
        # getting edge list
        edge_list = adj[0]
        edge_type = adj[1]

        if len(train_indices_nhop) == 0:
            edge_list_nhop = torch.LongTensor([[],[]])
            edge_type_nhop = torch.LongTensor([])
        else:
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)

        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()



        start = time.time()

        # self.entity_embeddings.data = F.normalize(
        #     self.entity_embeddings.data, p=2, dim=1).detach()

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        temp_entity_embeddings = self.entity_embeddings
        temp_relation_embeddings = self.relation_embeddings

        temp_entity_embeddings.data = F.normalize(temp_entity_embeddings.data, p=2, dim=1).detach()
        temp_relation_embeddings.data = F.normalize(temp_relation_embeddings.data, p=2, dim=1).detach()
        score_i = 0
        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(temp_entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0
        for i,layer in enumerate(self.sparse_gats):
            edge_embed = temp_relation_embeddings[edge_type]
            if i == 0:
                entities_upgraded = temp_entity_embeddings.mm(self.W_entities)
                relations_upgraded = temp_relation_embeddings.mm(self.W_entities)
                if -1 in self.score_layers:
                    self.record_entity_embeddings.data[:, score_i, :] = entities_upgraded.data
                    self.record_relation_embeddings.data[:, score_i, :] = relations_upgraded.data
                    score_i += 1
                temp_entity_embeddings_new, temp_relation_embeddings_new = layer(Corpus_, batch_inputs, temp_entity_embeddings,temp_relation_embeddings, edge_list, edge_type,edge_embed, edge_list_nhop, edge_type_nhop)
                temp_entity_embeddings = entities_upgraded + mask.unsqueeze(-1).expand_as(temp_entity_embeddings_new) * temp_entity_embeddings_new
                if self.not_res:
                    temp_relation_embeddings = temp_relation_embeddings_new
                else:
                    temp_relation_embeddings = relations_upgraded + temp_relation_embeddings_new
            else:
                temp_entity_embeddings_new, temp_relation_embeddings_new = layer(Corpus_, batch_inputs, temp_entity_embeddings, temp_relation_embeddings, edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)
                temp_entity_embeddings = temp_entity_embeddings + mask.unsqueeze(-1).expand_as(temp_entity_embeddings_new) * temp_entity_embeddings_new
                if self.not_res:
                    temp_relation_embeddings = temp_relation_embeddings_new
                else:
                    temp_relation_embeddings = temp_relation_embeddings + temp_relation_embeddings_new
            temp_entity_embeddings.data = F.normalize(temp_entity_embeddings.data, p=2, dim=1).detach()
            temp_relation_embeddings.data = F.normalize(temp_relation_embeddings.data, p=2, dim=1).detach()
            if (i in self.score_layers):
                self.record_entity_embeddings.data[:,score_i,:] = temp_entity_embeddings.data
                self.record_relation_embeddings.data[:,score_i,:] = temp_relation_embeddings.data
                score_i += 1


        if (self.depth_gat == 0) and (-1 in self.score_layers):
            entities_upgraded = temp_entity_embeddings.mm(self.W_entities)
            relations_upgraded = temp_relation_embeddings.mm(self.W_entities)
            self.record_entity_embeddings.data[:, score_i, :] = entities_upgraded.data
            self.record_relation_embeddings.data[:, score_i, :] = relations_upgraded.data
            temp_entity_embeddings = entities_upgraded
            temp_relation_embeddings = relations_upgraded

        self.final_entity_embeddings.data = temp_entity_embeddings.data
        self.final_relation_embeddings.data = temp_relation_embeddings.data

        return temp_entity_embeddings, temp_relation_embeddings
