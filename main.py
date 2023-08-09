import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm
import time 
import random
import numpy as np

import utils
from model import Model

import json


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, batch_save):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader) 
    batch_num = 0
    for idx, pos_1, pos_2, target in train_bar:
        batch_size = pos_1.shape[0]
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B] 
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature) 
        # neg
        mask = get_negative_mask(batch_size).cuda()
        # [2*B, 2*B-2]


        neg = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos_sim + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        elif estimator=='debias':
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos_sim + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')

        loss = (- torch.log(pos_sim /  (pos_sim + Ng))).mean()


        # get all image embeds 
        if batch_num == 0: 
            embeds_all = torch.cat((out_1, out_2), dim=1).detach() 
            index_all = idx.view(batch_size, -1).detach()
        else:
            embeds_all = torch.cat((embeds_all, torch.cat((out_1, out_2), dim=1)), dim=0).detach() 
            index_all = torch.cat((index_all, idx.view(batch_size, -1)), dim=0).detach()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        batch_num += 1
        batch_save += 1

        if batch_save != 0 and batch_save % args.save_batches == 0:
            torch.save(model.state_dict(), args.save_path+'/model_batch_{}.pth'.format(batch_save))
        if batch_save > batch_nums_all:
            return total_loss / total_num, embeds_all, index_all, batch_save
    return total_loss / total_num, embeds_all, index_all, batch_save


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for _, data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for _, data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100


# random -> topk
def build_graph(images_embeds):
    split_num = 20
    fold_len = images_embeds.size()[0] // split_num

    images_embeds = F.normalize(images_embeds, dim=-1)

    for i in range(split_num):
        start = i * fold_len
        if i == split_num -1:
            end = images_embeds.size()[0] 
        else:
            end = (i + 1) * fold_len 

        # random 
        neg_indices = torch.randint(0, images_embeds.shape[0], (end-start, args.M)).cuda()
        random_select_idx = neg_indices.view(1, -1)[0]

        anchor = images_embeds[start:end].unsqueeze(1)

        negs = images_embeds[random_select_idx].view(end-start, args.M, -1)

        B_score = torch.matmul(anchor, negs.permute(0,2,1)).squeeze()
        B_score = torch.exp(B_score / temperature)
        neg = B_score

        # top
        B_score, B_idx = neg.topk(k=edge_nums, dim=-1)
        B_indices = torch.gather(neg_indices, 1, B_idx)


        if i == 0:
            edge_score = B_score 
            adj_index = B_indices
        else:
            edge_score = torch.cat([edge_score, B_score], dim=0)
            adj_index = torch.cat([adj_index, B_indices], dim=0)

    edge_score = edge_score.detach().cpu() 
    adj_index = adj_index.detach().cpu()

    return map_index, adj_index, edge_score 



def sampling(start_node, adj_index, map_index, visit, restart_p):
    if args.sampling == 'RWR':
        walks, visit = random_walk(start_node, adj_index, map_index, visit, restart_p) 
    else:
        raise Exception("unvalid sampling method")
    return walks, visit


def random_walk(start_node, adj_index, map_index, visit, restart_p):
    # random walk + restart
    walks = set()
    walks.add(start_node)
    paths = [start_node]
    while len(walks) < batch_size:
        cur = paths[-1]
        nodes = adj_index[cur].tolist() 
        if len(nodes) > 0:
            if torch.rand(1).item() < restart_p: 
                paths.append(paths[0]) 
            else: 
                next_node = random.choice(nodes)
                paths.append(next_node) 
                if next_node not in visit:
                    walks.add(next_node) 
        else:
            break 
    visit = visit | walks 
    assert len(set(walks)) == batch_size
    return walks, visit



def generate_indices(graph, walks):
    map_index, adj_index, edge_weights = graph[0], graph[1].numpy(), graph[2] 
    indices = []
    for _ in range(args.update_batchs): 
        visit = set() 
        remain_images = list(set(map_index.keys()) - set(walks)) 
        if len(remain_images) == 0:
            return indices, walks
        start_node = random.choice(remain_images) 
        batch_idx, visit = sampling(start_node, adj_index, map_index, visit, restart_p=restart_p) 
        walks.extend(batch_idx) 
        indices.extend(batch_idx)
    return indices, walks

def update_graph(update_embeds, update_index, origin_embeds):
    map_index = {k:i for i, k in enumerate(update_index.view(1,-1)[0].numpy().tolist())}
    index = torch.tensor(list(map_index.keys()))
    embeds_images = update_embeds[torch.tensor(list(map_index.values()))]
    origin_embeds[index, :] = embeds_images 
    images_embeds = origin_embeds

    graph = build_graph(images_embeds) 
    return graph, images_embeds 

def get_embeds(embeds_all, index_all):
    embeds = torch.ones((len(train_data), embeds_all.shape[1]), dtype=torch.float).cuda()
    map_index = {k:i for i, k in enumerate(index_all.view(1,-1)[0].numpy().tolist())}

    index = torch.tensor(list(map_index.keys()))
    embeds_images = embeds_all[torch.tensor(list(map_index.values()))]
    embeds[index, :] = embeds_images 
    images_embeds = embeds 
    return map_index, images_embeds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset', default='stl10', type=str, help='train dataset')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--save_batches', default=1000, type=int, help='Number of batches to save checkpoint')
    parser.add_argument('--edge_nums', default=100, type=int, help='the number of edges for build graph')
    parser.add_argument('--restart_p', default=0.1, type=float, help='random walk with restart probability')
    parser.add_argument('--random_epochs', default=10, type=int, help='random shuflle epochs at beginning')
    parser.add_argument('--save_path', default='results', type=str, help='save path')
    parser.add_argument('--sampling', default='RW', type=str, help='sampling method')
    parser.add_argument('--M', default=1000, type=int, help='Number of edges in first selection')
    parser.add_argument('--update_batchs', default=100, type=int, help='graph update batchs')
    parser.add_argument('--estimator', default='hard', type=str, help='Choose loss function')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class priorx')
    parser.add_argument('--beta', default=1.0, type=float, help='Choose loss function')

    # args parse 
    args = parser.parse_args()
    print(args) 
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    with open(args.save_path + '/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    feature_dim, temperature, k, restart_p, edge_nums= args.feature_dim, args.temperature, args.k, args.restart_p, args.edge_nums
    batch_size, epochs = args.batch_size, args.epochs 
    estimator = args.estimator
    tau_plus = args.tau_plus
    beta = args.beta

    # data prepare 
    if args.dataset == 'stl10':
        train_data = utils.STL10Pair(root='data', split='train+unlabeled', transform=utils.train_transform_stl10, download=True)
        memory_data = utils.STL10Pair(root='data', split='train', transform=utils.test_transform_stl10, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_data = utils.STL10Pair(root='data', split='test', transform=utils.test_transform_stl10, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    else:
        raise Exception("unvalid dataset")
    
    args.save_batches = int(len(train_data)//batch_size+1) * 100

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    

    batch_nums_all = int(len(train_data)//batch_size+1) * args.epochs
    batch_save = 0
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        walks = [] 

        if epoch <= args.random_epochs:
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                    drop_last=False)
            train_loss, embeds_all, index_all, batch_save = train(model, train_loader, optimizer, batch_save)
            embeds_all, index_all = embeds_all.detach(), index_all.detach()
            # build graph
            if epoch == args.random_epochs:
                map_index, images_embeds = get_embeds(embeds_all, index_all)
                graph = build_graph(images_embeds)             
        else:
            t_generate_indices = time.time() 
            batch_num = len(train_data) // batch_size 
            cur_batch = 0
            # restart_p
            restart_p = args.restart_p - (epoch/5)*0.1
            if restart_p <= 0.05:
                restart_p = 0.05

            while len(set(walks)) < batch_num * batch_size:
                if cur_batch != 0 and cur_batch % args.update_batchs == 0:
                    # update graph
                    graph, images_embeds = update_graph(embeds_all, index_all, images_embeds) 
                indices, walks = generate_indices(graph, walks) 
                sampler = utils.Sampler(indices)
                train_loader = DataLoader(train_data,
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=1, pin_memory=True,
                                            sampler=sampler,
                                            ) 
                train_loss, embeds_all, index_all, batch_save = train(model, train_loader, optimizer, batch_save)
                embeds_all, index_all = embeds_all.detach(), index_all.detach()
                cur_batch += len(train_loader)
                if batch_save > batch_nums_all:
                    break

            if batch_save > batch_nums_all:
                break
            remain_images = list(set(map_index.keys()) - set(walks)) 
            walks.extend(remain_images)
            assert len(set(walks)) == len(train_data) 
            if len(remain_images) > 1:
                sampler = utils.Sampler(remain_images)
                train_loader = DataLoader(train_data,
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=1, pin_memory=True,
                                            sampler=sampler,
                                            ) 
                train_loss, embeds_all, index_all, batch_save = train(model, train_loader, optimizer, batch_save)
                embeds_all, index_all = embeds_all.detach(), index_all.detach()
            print("t_generate_indices=========", time.time() - t_generate_indices) 


        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(args.save_path+'/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(model.state_dict(), args.save_path+'/{}_model.pth'.format(save_name_pre))



