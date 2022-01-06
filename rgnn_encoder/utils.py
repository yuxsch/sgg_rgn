import torch
# from torchviz import make_dot, make_dot_from_trace

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CUDA = torch.cuda.is_available()


def save_model(model, name, epoch, folder_name):
    print("Saving Model")
    torch.save(model.state_dict(),
               (folder_name + "trained_{}.pth").format(epoch))
    print("Done saving Model")


gat_loss_func = nn.MarginRankingLoss(margin=0.5)

def print_grads(model):
    print(model.relation_embed.weight.grad)
    print(model.relation_gat_1.attention_0.a.grad)
    print(model.convKB.fc_layer.weight.grad)
    for name, param in model.named_parameters():
        print(name, param.grad)


def clip_gradients(model, gradient_clip_norm):
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, "norm before clipping is -> ", param.grad.norm())
            torch.nn.utils.clip_grad_norm_(param, args.gradient_clip_norm)
            print(name, "norm beafterfore clipping is -> ", param.grad.norm())


def plot_grad_flow(named_parameters, parameters):
    ave_grads = []
    max_grads = []
    layers = []

    for n, p in zip(named_parameters, parameters):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="r")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="g")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="g", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig('initial.png')
    plt.close()

def plot_grad_flow_low(named_parameters, parameters):
    ave_grads = []
    layers = []
    for n, p in zip(named_parameters, parameters):
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig('initial.png')
    plt.close()
