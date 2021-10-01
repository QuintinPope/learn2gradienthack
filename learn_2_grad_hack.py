#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load Omniglot, and
    * sample tasks and split them in adaptation and evaluation sets.
"""

import random
import torch
import learn2learn as l2l
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from torch import nn, optim


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(learner, loss, base_steps, batch_size, start_of_epoch_weight, preds_over_time_dir, save_figs, base_save_steps, device):
    adaptation_data = torch.rand(base_steps, batch_size, 1).to(device)
    adaptation_labels = torch.sin(adaptation_data * 10)
    evaluation_data = torch.rand(batch_size, 1).to(device)
    evaluation_labels = torch.ones(batch_size, 1).to(device)
    total_train_error = 0
    batch_predictions = []
    batch_data = []
    batch_labels = []
    base_losses = []
    valid_error = 0

    # Adapt the model
    for step in range(base_steps):
        #adaptation_data = torch.rand(batch_size, 1).to(device)
        #adaptation_labels = torch.sin(adaptation_data * 10)
        train_preds = learner(adaptation_data[step])
        train_error = loss(train_preds, adaptation_labels[step])
        learner.adapt(train_error)
        if step == 0:
            valid_error = valid_error + start_of_epoch_weight * loss(train_preds, evaluation_labels)
        base_losses.append(train_error.item())

        total_train_error += train_error.item() / base_steps
        if save_figs and (step % base_save_steps == base_save_steps - 1 or step == 0):
            plot_select = torch.zeros_like(adaptation_data[step]).bool()
            for i in range(0, len(plot_select), 5):
                plot_select[i] = True
            plt.plot(adaptation_data[step][plot_select].tolist(), train_preds[plot_select].tolist(), 'ro')
            plt.plot(adaptation_data[step][plot_select].tolist(), adaptation_labels[step][plot_select].tolist(), 'bo')
            plt.plot(adaptation_data[step][plot_select].tolist(), evaluation_labels[plot_select].tolist(), 'bo')

            plt.ylabel("Function Output")
            plt.xlabel("Function Input")
            plt.savefig(preds_over_time_dir + "base_learning_preds_epoch_" + str(step) + ".pdf", format="pdf")
            plt.cla()

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = (valid_error + loss(predictions, evaluation_labels)) / 2
    for x in train_preds:
        batch_predictions.append(x.item())
    for x in predictions:
        batch_predictions.append(x.item())
    for x in adaptation_data[0]:
        batch_data.append(x.item())
    for x in evaluation_data:
        batch_data.append(x.item())
    for x in adaptation_labels[0]:
        batch_labels.append(x.item())
    for x in evaluation_labels:
        batch_labels.append(x.item())
    return valid_error, total_train_error, batch_predictions, batch_data, batch_labels, base_losses

parser = argparse.ArgumentParser(description='Generates text counterfactuals using the CLOSS method.')


parser.add_argument("--output_path", help="Path to store output")
parser.add_argument("--preface", help="Preface to add to output")
parser.add_argument("--meta_lr", help="LR for the meta learning optimizer", type=float)
parser.add_argument("--meta_momentum", help="momentum for meta learning optimizer", type=float)
parser.add_argument("--optim", help="Optimizer for meta learning. Default is SGD", default="SGD", type=str)
parser.add_argument("--base_lr", help="LR for the base optimizer", type=float)
parser.add_argument("--min_lr", help="Min LR for optimizer schedule. If not specified, no scheduler is used", default=0.0, type=float)
parser.add_argument("--max_lr", help="Max LR for optimizer schedule", default=0.0, type=float)
parser.add_argument("--meta_batch_size", help="Batch size for meta learning optimizer", type=int)
parser.add_argument("--batch_size", help="Batch size for base optimizer", type=int)
parser.add_argument("--base_steps", help="Steps to run base optimizer for each pass of meta optimization. Can also give a range, i.e., \"100,200\"", type=str)
parser.add_argument("--meta_steps", help="Total steps of meta optimization", type=int)
parser.add_argument("--cuda", help="Whether to use cuda", type=bool)
parser.add_argument("--seed", help="Random seed", type=int)
parser.add_argument("--start_of_epoch_weight", help="Weight of meta learning loss from predictions during start of base learning epoch", type=float)
parser.add_argument("--save_steps", help="Number of steps between saving the model and recording plots", type=int)
parser.add_argument("--base_save_steps", help="Number of steps between recording plots during base model training", type=int)
parser.add_argument("--load_model", help="Path to load model from if starting from a checkpoint (optional)", default=None, type=str)
parser.add_argument("--resume_epoch", help="Epoch of meta optimization to start counting from (optional)", default=0, type=int)
parser.add_argument("--sizes", help="Sizes of fully connected network (optional). Encoded as CSV string. Default is \"256,128,64,64\"", default=None, type=str)
parser.add_argument("--loss_function", help="Loss function (optional). Default is l1", default="l1", type=str)
parser.add_argument("--l2_penalty", help="L2 penalty for the meta learning optimizer (optional, default 0)", default=0.0, type=float)


args = parser.parse_args()

output_path = args.output_path
preface = args.preface
meta_lr = args.meta_lr
meta_momentum = args.meta_momentum
base_lr = args.base_lr
min_lr = args.min_lr
max_lr = args.max_lr
meta_batch_size = args.meta_batch_size
batch_size = args.batch_size
base_steps = args.base_steps
meta_steps = args.meta_steps
cuda = args.cuda
seed = args.seed
start_of_epoch_weight = args.start_of_epoch_weight
save_steps = args.save_steps
base_save_steps = args.base_save_steps
load_model = args.load_model
resume_epoch = args.resume_epoch
sizes = args.sizes
loss_function = str.lower(args.loss_function)
meta_optimizer = str.lower(args.optim)
l2_penalty = args.l2_penalty

base_steps = base_steps.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
base_steps = [int(s) for s in base_steps.split(",")]
if not sizes is None:
    sizes = sizes.replace(" ", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    sizes = [int(s) for s in sizes.split(",")]
if output_path[-1] != '/':
    output_path = output_path + '/'

f = open(output_path + preface + '_args_record.txt', "w")
f.write("output_path:" + output_path + " preface: " + preface + " meta_lr: " + str(meta_lr) + " meta_momentum: " + str(meta_momentum))
f.write(" base_lr: " + str(base_lr) + " meta_batch_size: " + str(meta_batch_size) + " batch_size: " + str(batch_size))
f.write(" base_steps: " + str(base_steps) + " meta_steps: " + str(meta_steps) + " cuda: " + str(cuda) + " seed: " + str(seed))
f.write(" start_of_epoch_weight: " + str(start_of_epoch_weight) + " save_steps: " + str(save_steps) + " base_save_steps: " + str(base_save_steps))
f.write(" load_model: " + str(load_model) + " resume_epoch: " + str(resume_epoch) + " sizes: " + str(sizes) + " loss_function: " + str(loss_function))
f.write(" meta_optimizer: " + str(meta_optimizer) + " min_lr: " + str(min_lr) + " max_lr: " + str(max_lr) + " l2_penalty: " + str(l2_penalty))
f.close()

random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cpu')
if cuda:
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda')

# Create model
if load_model is None:
    model = l2l.vision.models.OmniglotFC(1, 1, sizes=sizes)
else:
    model = torch.load(load_model)
model.to(device)
print(model)

maml = l2l.algorithms.MAML(model, lr=base_lr, first_order=False)
if meta_optimizer == 'sgd':
    opt = optim.SGD(maml.parameters(), meta_lr, meta_momentum, weight_decay=l2_penalty)
elif meta_optimizer == 'adam':
    opt = optim.Adam(maml.parameters(), meta_lr, weight_decay=l2_penalty)
elif meta_optimizer == 'rprop':
    opt = optim.Rprop(maml.parameters(), meta_lr, etas=(0.5, 1.2), weight_decay=l2_penalty)
elif meta_optimizer == 'adadelta':
    opt = optim.Adadelta(maml.parameters(), meta_lr, weight_decay=l2_penalty)
elif meta_optimizer == 'adagrad':
    opt = optim.Adagrad(maml.parameters(), meta_lr, weight_decay=l2_penalty)
elif meta_optimizer == 'lbfgs':
    opt = optim.LBFGS(maml.parameters(), meta_lr, weight_decay=l2_penalty)
else:
    print("Error: invalid optim:", meta_optimizer)

if min_lr > 0:
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=min_lr, max_lr=max_lr, cycle_momentum=False, step_size_up=100)

if loss_function == 'l1':
    loss = nn.L1Loss(reduction='mean')
elif loss_function == 'l2':
    loss = nn.MSELoss(reduction='mean')
else:
    print("Invalid loss function:", loss_function)

all_meta_losses = []
all_train_losses = []
for iteration in range(resume_epoch, meta_steps + resume_epoch):
    opt.zero_grad()
    meta_train_error = 0.0
    batch_train_error = 0.0
    all_data = []
    all_predictions = []
    all_labels = []
    all_base_losses = []
    save_figs = False
    preds_over_time_dir = output_path + preface + "_preds_over_time_epoch_" + str(iteration) + '/'
    if iteration % save_steps == save_steps - 1 or iteration == resume_epoch:
        os.mkdir(preds_over_time_dir)
        save_figs = True
    for task in range(meta_batch_size):
        # Compute meta-training loss
        learner = maml.clone()
        save_figs = save_figs and (task == 0)
        if len(base_steps) > 1:
            steps = torch.randint(low=base_steps[0], high=base_steps[1] + 1, size=(1,)).item()
        else:
            steps = base_steps[0]
        evaluation_error, train_error, batch_predictions, batch_data, batch_labels, base_losses = fast_adapt(learner,
                                                                                                             loss,
                                                                                                             steps,
                                                                                                             batch_size,
                                                                                                             start_of_epoch_weight,
                                                                                                             preds_over_time_dir,
                                                                                                             save_figs,
                                                                                                             base_save_steps,
                                                                                                             device)
        evaluation_error.backward()
        meta_train_error += evaluation_error.item()
        batch_train_error += train_error
        for i in range(0, len(batch_predictions), 16):
            all_predictions.append(batch_predictions[i])
            all_data.append(batch_data[i])
            all_labels.append(batch_labels[i])
        for i in range(0, len(base_losses), 16):
            all_base_losses.append(base_losses[i])
        all_train_losses.append(train_error)
        all_meta_losses.append(evaluation_error.item())

    # Print some metrics
    print('\n')
    print('Epoch           ', iteration)
    print('Meta Train Error', meta_train_error / meta_batch_size)
    print('Base Train Error', batch_train_error / meta_batch_size)

    # Average the accumulated gradients and optimize
    for p in maml.parameters():
        p.grad.data.mul_(1.0 / meta_batch_size)
    opt.step()
    if min_lr > 0:
        scheduler.step()

    if iteration % save_steps == save_steps - 1 or iteration == resume_epoch:
        warm_up = 0 #max(len(all_data) - 3000, 0)
        end = len(all_data) - 0
        torch.save(model, output_path + preface + "_epoch_" + str(iteration) + ".pth")
        plt.plot(all_data[warm_up:end], all_predictions[warm_up:end], 'ro')
        plt.plot(all_data[warm_up:end], all_labels[warm_up:end], 'bo')
        plt.xlabel("Function Input")
        plt.ylabel("Function Output")
        plt.savefig(output_path + preface + "_predictions_epoch_" + str(iteration) + ".pdf", format="pdf")
        plt.cla()
        plt.show()

        plt.plot([i for i in range(len(all_base_losses))], all_base_losses)
        plt.ylabel("Loss")
        plt.xlabel("Base Opt Steps")
        plt.savefig(output_path + preface + "_base_losses_epoch_" + str(iteration) + ".pdf", format="pdf")
        plt.cla()
        plt.show()

meta_losses_df = pd.DataFrame(all_meta_losses)
base_losses_df = pd.DataFrame(all_train_losses)
meta_losses_df.to_csv(output_path + preface + "_meta_losses.csv", sep=',')
base_losses_df.to_csv(output_path + preface + "_base_losses.csv", sep=',')

