import torch

from models import SpKBGATModified
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from copy import deepcopy
from preprocess import init_embeddings, build_data
from create_batch import Corpus
from utils import save_model
import random
import argparse
import os
import time
import pickle
from na_txt_generator import generate_entity, generate_relation

def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/VG-KB/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=2000, help="Number of epochs")
    args.add_argument("-d_g", "--depth_gat", type=int,
                      default=3, help="depth of gat")
    args.add_argument("-s_layer", "--score_layers", type=int, nargs='+',
                      default=[], help="Score_layers")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=True, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=200, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-4)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=False)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=False)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=False)
    args.add_argument("-eval_type", "--eval_type", default='last')
    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
    args.add_argument("-out_dim", "--entity_out_dim", type=int,
                      default=100, help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int,
                      default=2, help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    args.add_argument("-g", "--gpu", type=int,
                      default=-1, help="GPU num, default -1")
    args.add_argument("-tc", "--test_cpu", type=bool, default=False)
    args.add_argument("-nr", "--not_res", type=bool, default=False)
    args = args.parse_args()
    return args


args = parse_args()
print(args)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)


print('Use CUDA? :' + str(use_cuda))
# %%


def load_data(args):
    train_data, validation_data, test_data, entity2id, relation2id, headTailSelector, unique_entities_train = build_data(
        args.data, is_unweigted=False, directed=True)

    if args.pretrained_emb:
        entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
                                                                 os.path.join(args.data, 'relation2vec.txt'))
        print("Initialised relations and entities from TransE")

    else:
        entity_embeddings = np.random.randn(
            len(entity2id), args.embedding_size)
        relation_embeddings = np.random.randn(
            len(relation2id), args.embedding_size)
        print("Initialised relations and entities randomly")

    corpus = Corpus(args, train_data, validation_data, test_data, entity2id, relation2id, headTailSelector,
                    args.batch_size_gat, args.valid_invalid_ratio_gat, unique_entities_train, args.get_2hop)

    return corpus, torch.FloatTensor(entity_embeddings), torch.FloatTensor(relation_embeddings)


Corpus_, entity_embeddings, relation_embeddings = load_data(args)


if(args.get_2hop):
    file = args.data + "/2hop.pickle"
    with open(file, 'wb') as handle:
        pickle.dump(Corpus_.node_neighbors_2hop, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if(args.use_2hop):
    print("Opening node_neighbors pickle object")
    file = args.data + "/2hop.pickle"
    with open(file, 'rb') as handle:
        node_neighbors_2hop = pickle.load(handle)

entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
# %%



def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(
        train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss


def train_gat(args):

    # Creating the gat model here.
    ####################################

    print("Defining model")

    print(
        "\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT))
    model_gat = SpKBGATModified(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT, args.depth_gat, args.score_layers, args.not_res)

    if use_cuda:
        model_gat.cuda()

    optimizer = torch.optim.Adam(
        model_gat.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.6, last_epoch=-1)

    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)


    if(args.use_2hop):
        # print('test')
        # print(node_neighbors_2hop)

        current_batch_2hop_indices = Corpus_.get_batch_nhop_neighbors_all(args, Corpus_.unique_entities_train, node_neighbors_2hop)
        # print(current_batch_2hop_indices)
        if use_cuda:
            current_batch_2hop_indices = Variable(
                torch.LongTensor(current_batch_2hop_indices)).cuda()
        else:
            current_batch_2hop_indices = Variable(
                torch.LongTensor(current_batch_2hop_indices))
    else:
        if use_cuda:
            current_batch_2hop_indices = Variable(torch.LongTensor([])).cuda()
        else:
            current_batch_2hop_indices = Variable(torch.LongTensor([]))

    epoch_losses = []   # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))

    for epoch in range(args.epochs_gat):
        print("\nepoch-> ", epoch)
        random.shuffle(Corpus_.train_triples)
        Corpus_.train_indices = np.array(
            list(Corpus_.train_triples)).astype(np.int32)

        model_gat.train()  # getting in training mode
        start_time = time.time()
        epoch_loss = []

        if len(Corpus_.train_indices) % args.batch_size_gat == 0:
            num_iters_per_epoch = len(
                Corpus_.train_indices) // args.batch_size_gat
        else:
            num_iters_per_epoch = (
                len(Corpus_.train_indices) // args.batch_size_gat) + 1

        for iters in range(num_iters_per_epoch):
            start_time_iter = time.time()
            train_indices, train_values = Corpus_.get_iteration_batch(iters)

            if use_cuda:
                train_indices = Variable(
                    torch.LongTensor(train_indices)).cuda()
                train_values = Variable(torch.FloatTensor(train_values)).cuda()

            else:
                train_indices = Variable(torch.LongTensor(train_indices))
                train_values = Variable(torch.FloatTensor(train_values))

            # forward pass
            entity_embed, relation_embed = model_gat(
                Corpus_, Corpus_.train_adj_matrix, train_indices, current_batch_2hop_indices)

            optimizer.zero_grad()

            loss = batch_gat_loss(
                gat_loss_func, train_indices, entity_embed, relation_embed)

            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data.item())

            end_time_iter = time.time()

            print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(
                iters, end_time_iter - start_time_iter, loss.data.item()))

        scheduler.step()
        print("Epoch {} , average loss {} , epoch_time {}".format(
            epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
        epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

        if epoch == args.epochs_gat - 1:
            save_model(model_gat, args.data, epoch, args.output_folder + str(args.gpu) + '/gat/')
            np.save(args.output_folder + str(args.gpu) + '/gat/' + 'loss' + '.npy', np.array(epoch_losses))

    t = model_gat.final_entity_embeddings.data.cpu()
    r = model_gat.final_relation_embeddings.data.cpu()
    generate_entity(t,'na')
    generate_relation(r,'na')

train_gat(args)
