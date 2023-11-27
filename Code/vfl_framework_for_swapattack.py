import argparse
import ast
import copy
import os
import random
import sys
import time
from time import time

import dill
from colorama import Back, Fore, Style, init

sys.path.insert(0, "./")

import matplotlib.pyplot as plt
import my_optimizers
import numpy as np
import pandas as pd
import possible_defenses
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from datasets import get_dataset
from models import model_sets
from my_utils import utils
from sklearn.manifold import TSNE
from torch.utils.data.sampler import SequentialSampler
from utils_for_swapAttack import judge_converge

plt.switch_backend("agg")

D_ = 2**13
BATCH_SIZE = 1000


class MySequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)


class VflFramework(nn.Module):
    def __init__(self, setting_str=None):
        super(VflFramework, self).__init__()
        self.setting_str = setting_str  # counter for direct label inference attack
        self.train_loader = None  # set for swap attack.
        # get num_classes
        dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
        self.num_classes = dataset_setup.num_classes
        self.inferred_correct = 0
        self.inferred_wrong = 0
        # bottom model a can collect output_a and grads for label inference attack
        self.collect_outputs_a = False
        # self.outputs_a = torch.tensor([]).cuda()
        self.outputs_a = None
        self.grads_a = None
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()
        self.is_model_output_correct = None
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.if_collect_training_dataset_labels = False

        # adversarial options
        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad
        # self.defense_ss = args.ss

        # indicates whether to conduct the direct label inference attack
        self.direct_attack_on = False

        # loss funcs
        self.loss_func_top_model = nn.CrossEntropyLoss()
        self.loss_func_bottom_model = utils.keep_predict_loss

        # bottom model A
        self.malicious_bottom_model_a = model_sets.BottomModel(
            dataset_name=args.dataset
        ).get_model(half=args.half, is_adversary=True)
        # bottom model B
        self.benign_bottom_model_b = model_sets.BottomModel(
            dataset_name=args.dataset
        ).get_model(half=args.half, is_adversary=False)
        # top model
        self.top_model = model_sets.TopModel(dataset_name=args.dataset).get_model()

        # This setting is for adversarial experiments except sign SGD
        if args.use_mal_optim_top:
            self.optimizer_top_model = my_optimizers.MaliciousSGD(
                self.top_model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer_top_model = optim.SGD(
                self.top_model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        if args.dataset != "Yahoo":
            if args.use_mal_optim:
                self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(
                    self.malicious_bottom_model_a.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            else:
                self.optimizer_malicious_bottom_model_a = optim.SGD(
                    self.malicious_bottom_model_a.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            if args.use_mal_optim_all:
                self.optimizer_benign_bottom_model_b = my_optimizers.MaliciousSGD(
                    self.benign_bottom_model_b.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            else:
                self.optimizer_benign_bottom_model_b = optim.SGD(
                    self.benign_bottom_model_b.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
        else:
            if args.use_mal_optim:
                self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(
                    [
                        {
                            "params": self.malicious_bottom_model_a.mixtext_model.bert.parameters(),
                            "lr": 5e-6,
                        },
                        {
                            "params": self.malicious_bottom_model_a.mixtext_model.linear.parameters(),
                            "lr": 5e-4,
                        },
                    ],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            else:
                self.optimizer_malicious_bottom_model_a = optim.SGD(
                    [
                        {
                            "params": self.malicious_bottom_model_a.mixtext_model.bert.parameters(),
                            "lr": 5e-6,
                        },
                        {
                            "params": self.malicious_bottom_model_a.mixtext_model.linear.parameters(),
                            "lr": 5e-4,
                        },
                    ],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            if args.use_mal_optim_all:
                self.optimizer_benign_bottom_model_b = my_optimizers.MaliciousSGD(
                    [
                        {
                            "params": self.benign_bottom_model_b.mixtext_model.bert.parameters(),
                            "lr": 5e-6,
                        },
                        {
                            "params": self.benign_bottom_model_b.mixtext_model.linear.parameters(),
                            "lr": 5e-4,
                        },
                    ],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )
            else:
                self.optimizer_benign_bottom_model_b = optim.SGD(
                    [
                        {
                            "params": self.benign_bottom_model_b.mixtext_model.bert.parameters(),
                            "lr": 5e-6,
                        },
                        {
                            "params": self.benign_bottom_model_b.mixtext_model.linear.parameters(),
                            "lr": 5e-4,
                        },
                    ],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                )

    def simulate_train_epoch(
        self,
        train_loader,
        origin_sample_ids,
        schedulers,
        epoch,
        is_swapAttack=False,
        target_label=-1,
    ):
        if epoch >= args.epochs:
            return

        self.train_loader = train_loader
        print(
            "model.optimizer_top_model current lr {:.5e}".format(
                self.optimizer_top_model.param_groups[0]["lr"]
            )
        )
        print(
            "model.optimizer_malicious_bottom_model_a current lr {:.5e}".format(
                self.optimizer_malicious_bottom_model_a.param_groups[0]["lr"]
            )
        )
        print(
            "model.optimizer_benign_bottom_model_b current lr {:.5e}".format(
                self.optimizer_benign_bottom_model_b.param_groups[0]["lr"]
            )
        )

        if self.outputs_a is None:
            data = [
                [-1.0 for _ in range(self.num_classes)]
                for _ in range(len(origin_sample_ids))
            ]
            self.outputs_a = torch.tensor(data).to(torch.float).cuda()
            self.grads_a = torch.tensor(data).to(torch.float).cuda()
            self.is_model_output_correct = (
                torch.tensor([False for _ in range(len(origin_sample_ids))])
                .to(torch.bool)
                .cuda()
            )

        for batch_id, (data, target) in enumerate(train_loader):
            # move data to GPU.
            if args.dataset == "Yahoo":
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            else:
                data = data.float().cuda()
                target = target.long().cuda()

            start = batch_id * args.batch_size
            end = min((batch_id + 1) * args.batch_size, len(origin_sample_ids))
            batch_sample_idxes = origin_sample_ids[start:end]

            # cal loss per batch
            loss_framework = self._simulate_train_batch(
                data,
                target,
                batch_id,
                batch_sample_idxes,
                is_swapAttack=is_swapAttack,
                target_label=target_label,
            )

            if batch_id % 25 == 0:
                if args.dataset == "Criteo":
                    num_samples = len(train_loader) * BATCH_SIZE
                else:
                    num_samples = len(train_loader.dataset)
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_id * len(data),
                        num_samples,
                        100.0 * batch_id / len(train_loader),
                        loss_framework.data.item(),
                    )
                )

        for scheduler in schedulers:
            scheduler.step()

    def _simulate_train_batch(
        self,
        data,
        target,
        batch_id,
        batch_sample_ids,
        is_swapAttack=False,
        target_label=None,
    ):
        timer_mal = 0
        timer_benign = 0
        # simulate: bottom models forward, top model forward, top model backward and update, bottom backward and update

        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        if self.if_collect_training_dataset_labels:
            self.labels_training_dataset = torch.cat(
                (self.labels_training_dataset, target), dim=0
            )

        # store grad of input of top model/outputs of bottom models
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)

        # --bottom models forward--
        x_a, x_b = self.split_data(data)

        # swap attack here
        if is_swapAttack:
            # change x_a
            target_batch_ids = self.target_batch_ids[batch_id]
            target_sample_ids = self.target_sample_ids[batch_id]
            x_a = self._swap_data(x_a, target_batch_ids, target_label)

        # -bottom model A-
        self.malicious_bottom_model_a.train(mode=True)
        start = time()
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        end = time()
        time_cost = end - start
        timer_mal += time_cost

        # -bottom model B-
        self.benign_bottom_model_b.train(mode=True)
        start2 = time()
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)
        end2 = time()
        time_cost2 = end2 - start2
        timer_benign += time_cost2

        # -top model-
        # (we omit interactive layer for it doesn't effect our attack or possible defenses)
        # by concatenating output of bottom a/b(dim=10+10=20), we get input of top model
        input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data

        if args.use_top_model:
            self.top_model.train(mode=True)
            output_framework = self.top_model(
                input_tensor_top_model_a, input_tensor_top_model_b
            )
            # --top model backward/update--
            loss_framework = model_sets.update_top_model_one_batch(
                optimizer=self.optimizer_top_model,
                model=self.top_model,
                output=output_framework,
                batch_target=target,
                loss_func=self.loss_func_top_model,
            )
        else:
            output_framework = input_tensor_top_model_a + input_tensor_top_model_b
            loss_framework = self.loss_func_top_model(output_framework, target)
            loss_framework.backward()

        # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
        grad_output_bottom_model_a = input_tensor_top_model_a.grad
        grad_output_bottom_model_b = input_tensor_top_model_b.grad

        # bottom model a can collect output_a and grads_a for label inference attack
        # clone will cope the message about grad.
        if not is_swapAttack:
            self.outputs_a[batch_sample_ids] = output_tensor_bottom_model_a.clone()
            self.grads_a[batch_sample_ids] = grad_output_bottom_model_a.detach()
        else:
            if len(target_batch_ids) > 0:
                for i, id in enumerate(target_batch_ids):
                    l2_norm_grad = torch.norm(
                        grad_output_bottom_model_a[id].detach(), p=2
                    ).item()
                    self.swap_grads[batch_id][i].append(l2_norm_grad)

                    l2_norm_grad_dis = torch.norm(
                        grad_output_bottom_model_a[id].detach()
                        - self.grads_a[target_sample_ids[i]],
                        p=2,
                    ).item()
                    self.swap_grads_dis[batch_id][i].append(l2_norm_grad_dis)

        # defenses here: the server(who controls top model) can defend against label inference attack by protecting
        # print("before defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)
        # gradients sent to bottom models
        model_all_layers_grads_list = [
            grad_output_bottom_model_a,
            grad_output_bottom_model_b,
        ]
        # privacy preserving deep learning
        if self.defense_ppdl:
            possible_defenses.dp_gc_ppdl(
                epsilon=1.8,
                sensitivity=1,
                layer_grad_list=[grad_output_bottom_model_a],
                theta_u=args.ppdl_theta_u,
                gamma=0.001,
                tau=0.0001,
            )
            possible_defenses.dp_gc_ppdl(
                epsilon=1.8,
                sensitivity=1,
                layer_grad_list=[grad_output_bottom_model_b],
                theta_u=args.ppdl_theta_u,
                gamma=0.001,
                tau=0.0001,
            )
        # gradient compression
        if self.defense_gc:
            tensor_pruner = possible_defenses.TensorPruner(
                zip_percent=args.gc_preserved_percent
            )
            for tensor_id in range(len(model_all_layers_grads_list)):
                tensor_pruner.update_thresh_hold(model_all_layers_grads_list[tensor_id])
                # print("tensor_pruner.thresh_hold:", tensor_pruner.thresh_hold)
                model_all_layers_grads_list[tensor_id] = tensor_pruner.prune_tensor(
                    model_all_layers_grads_list[tensor_id]
                )
        # differential privacy
        if self.defense_lap_noise:
            dp = possible_defenses.DPLaplacianNoiseApplyer(beta=args.noise_scale)
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = dp.laplace_mech(
                    model_all_layers_grads_list[tensor_id]
                )
        # multistep gradient
        if self.defense_multistep_grad:
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[
                    tensor_id
                ] = possible_defenses.multistep_gradient(
                    model_all_layers_grads_list[tensor_id],
                    bins_num=args.multistep_grad_bins,
                    bound_abs=args.multistep_grad_bound_abs,
                )
        # sign SGD
        # if self.defense_ss:
        #     for tensor in model_all_layers_grads_list:
        #         torch.sign(tensor, out=tensor)
        grad_output_bottom_model_a, grad_output_bottom_model_b = tuple(
            model_all_layers_grads_list
        )
        # print("after defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)

        # --bottom models backward/update--
        # -bottom model a: backward/update-
        # print("malicious_bottom_model_a")
        if not is_swapAttack:
            # not update during swapAttack
            start = time()
            model_sets.update_bottom_model_one_batch(
                optimizer=self.optimizer_malicious_bottom_model_a,
                model=self.malicious_bottom_model_a,
                output=output_tensor_bottom_model_a,
                batch_target=grad_output_bottom_model_a,
                loss_func=self.loss_func_bottom_model,
            )
            end = time()
            time_cost = end - start
            timer_mal += time_cost
        # -bottom model b: backward/update-
        # print("benign_bottom_model_b")
        model_sets.update_bottom_model_one_batch(
            optimizer=self.optimizer_benign_bottom_model_b,
            model=self.benign_bottom_model_b,
            output=output_tensor_bottom_model_b,
            batch_target=grad_output_bottom_model_b,
            loss_func=self.loss_func_bottom_model,
        )
        end2 = time()
        time_cost2 = end2 - end
        timer_benign += time_cost2
        timer_on = False
        if timer_on:
            print("timer_mal:", timer_mal)
            print("timer_benign:", timer_benign)

        return loss_framework

    def _swap_data(self, x_a, target_batch_ids, target_label):
        """x_a: batch data of participant a, tensor"""
        if len(target_batch_ids) == 0:
            return x_a

        full_feat = self._get_labeled_feat(target_label)
        swapped_feat = full_feat[:, :, 0 : args.half]
        for id in target_batch_ids:
            x_a[id] = torch.clone(swapped_feat)

        return x_a

    def _get_labeled_feat(self, target_label):
        """Select the sample with the smallest gradient norm among multiple labeled samples."""
        candidate_ids = self.labeled_sample_ids[target_label]
        if len(candidate_ids) == 1:
            target_sample_id = candidate_ids[0]
        else:
            l2_norm_grads = torch.norm(self.grads_a[candidate_ids], dim=1)
            target_sample_id = candidate_ids[torch.argmin(l2_norm_grads)]

        feat, _ = self.train_loader.dataset[target_sample_id]
        return feat

    def gen_labeled_samples(self, train_loader):
        """Each class generates a specified number of labeled samples,
        and we assume that the labeled samples are the ones for which the model fits correctly.
        Therefore, we select here based on the model prediction results and the norm of the gradient.
        """
        targets = train_loader.dataset.targets
        # Mark whether the sample has been attacked
        self.attacked_sample_ids = torch.tensor([False for _ in range(len(targets))])

        self.labeled_sample_ids = [[] for _ in range(self.num_classes)]
        labeled_num = args.labeled_perclass

        # sort according to gradient from small to large
        l2_norm_per_row = torch.norm(self.grads_a, dim=1)
        sorted_values, sorted_indices = torch.sort(l2_norm_per_row)

        count = self.num_classes * labeled_num
        for l2_norm, id in zip(sorted_values, sorted_indices):
            if self.is_model_output_correct[id]:
                class_id = targets[id]
                if len(self.labeled_sample_ids[class_id]) < labeled_num:
                    self.labeled_sample_ids[class_id].append(id.item())
                    self.attacked_sample_ids[id] = True
                    count -= 1

            if count == 0:
                break

        print(
            f"{Style.BRIGHT}{Fore.RED}Labeled samples are generated: {self.labeled_sample_ids}{Style.RESET_ALL}"
        )
        # print("Predict result:", self.is_model_output_correct[torch.tensor(self.labeled_sample_ids).flatten()].tolist())
        print(
            "Grads indices:",
            l2_norm_per_row[torch.tensor(self.labeled_sample_ids).flatten()].tolist(),
        )
        print("Max grads:", sorted_values[-10:])

    def gen_target_ids(self, origin_sample_ids):
        """
        Generate an index of samples to be attacked for the current attack period.
        Select target samples based on gradient norm.
        """
        print("Start generate targets ids per attack period.")
        l2_norm_samples = torch.norm(self.grads_a, dim=1)
        self.target_sample_ids, self.target_batch_ids, self.swap_grads = [], [], []

        start, end = 0, args.batch_size
        while start <= end:
            batch_sample_ids = origin_sample_ids[start:end]
            l2_norm_batch = l2_norm_samples[batch_sample_ids]
            sorted_indices = torch.argsort(l2_norm_batch)

            target_sample_ids_b, target_batch_ids_b, swap_grads_b = [], [], []
            for batch_id in sorted_indices:
                sample_id = batch_sample_ids[batch_id]

                if not self.attacked_sample_ids[sample_id]:
                    target_sample_ids_b.append(sample_id)
                    target_batch_ids_b.append(batch_id)
                    swap_grads_b.append([])

                if len(target_sample_ids_b) == args.batch_swap_size:
                    break

            self.target_sample_ids.append(target_sample_ids_b)
            self.target_batch_ids.append(target_batch_ids_b)
            self.swap_grads.append(swap_grads_b)

            start += args.batch_size
            end += args.batch_size
            end = min(end, len(origin_sample_ids))

        self.swap_grads_dis = copy.deepcopy(self.swap_grads)
        print("Generate targets ids done.")
        return

    def test(
        self,
        test_loader,
        k=5,
        loss_func_top_model=None,
        is_train_loader=False,
        origin_sample_ids=None,
        save_to_file=False,
        file_path="",
    ):
        test_loss = 0
        correct_top1 = 0
        correct_topk = 0
        count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # for data, target in test_loader:
                if args.dataset == "Yahoo":
                    for i in range(len(data)):
                        data[i] = data[i].long().cuda()
                    target = target[0].long().cuda()
                else:
                    data = data.float().cuda()
                    target = target.long().cuda()

                # set all sub-models to eval mode.
                self.malicious_bottom_model_a.eval()
                self.benign_bottom_model_b.eval()
                self.top_model.eval()

                # run forward process of the whole framework
                x_a, x_b = self.split_data(data)
                output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
                output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)

                if args.use_top_model:
                    output_framework = self.top_model(
                        output_tensor_bottom_model_a, output_tensor_bottom_model_b
                    )
                else:
                    output_framework = (
                        output_tensor_bottom_model_a + output_tensor_bottom_model_b
                    )

                correct_top1_batch, correct_topk_batch = self.correct_counter(
                    output_framework, target, (1, k)
                )
                output_framework_clone = output_framework.detach()
                # sum up batch loss
                test_loss += loss_func_top_model(output_framework, target).data.item()

                correct_top1 += correct_top1_batch
                correct_topk += correct_topk_batch

                if args.dataset == "Criteo" and count == test_loader.train_batches_num:
                    break

                # record output
                if is_train_loader:
                    start = batch_idx * args.batch_size
                    end = min((batch_idx + 1) * args.batch_size, len(origin_sample_ids))
                    batch_sample_idxes = origin_sample_ids[start:end]

                    _, pred = output_framework_clone.topk(
                        1, dim=1, largest=True, sorted=True
                    )
                    is_correct = torch.eq(pred, target.view(-1, 1)).flatten()
                    self.is_model_output_correct[batch_sample_idxes] = is_correct

            if args.dataset == "Criteo":
                num_samples = len(test_loader) * BATCH_SIZE
            else:
                num_samples = len(test_loader.dataset)
            test_loss /= num_samples
            text = "Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct_top1,
                num_samples,
                100.00 * float(correct_top1) / num_samples,
                k,
                correct_topk,
                num_samples,
                100.00 * float(correct_topk) / num_samples,
            )
            self.print_txtAndStd(save_to_file, text, file_path)

    def eval(
        self,
        train_loader,
        val_loader,
        origin_sample_ids,
        save_to_file=False,
        file_path="",
    ):
        """eval model after each train."""
        text = "Evaluation on the training dataset:"
        self.print_txtAndStd(save_to_file, text, file_path)
        self.test(
            test_loader=train_loader,
            k=args.k,
            loss_func_top_model=self.loss_func_top_model,
            is_train_loader=True,
            origin_sample_ids=origin_sample_ids,
            save_to_file=save_to_file,
            file_path=file_path,
        )
        text = "Evaluation on the testing dataset:"
        self.print_txtAndStd(save_to_file, text, file_path)
        self.test(
            test_loader=val_loader,
            k=args.k,
            loss_func_top_model=self.loss_func_top_model,
            save_to_file=save_to_file,
            file_path=file_path,
        )

    def forward(self, x):
        # in vertical federated setting, each party has non-lapping features of the same sample
        x_a, x_b = self.split_data(x)
        out_a = self.malicious_bottom_model_a(x_a)
        out_b = self.benign_bottom_model_b(x_b)
        if args.use_top_model:
            out = self.top_model(out_a, out_b)
        else:
            out = out_a + out_b
        return out

    @staticmethod
    def print_txtAndStd(save_to_file, text, file_path):
        if save_to_file:
            savedStdout = sys.stdout
            with open(file_path, "a+") as file:
                sys.stdout = file
                print(text)
                sys.stdout = savedStdout
        print(text)

    @staticmethod
    def correct_counter(output, target, topk=(1, 5)):
        correct_counts = []
        for k in topk:
            _, pred = output.topk(k, 1, True, True)
            correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
            correct_counts.append(correct_k)
        return correct_counts

    @staticmethod
    def split_data(data):
        if args.dataset == "Yahoo":
            x_b = data[1]
            x_a = data[0]
        elif args.dataset in ["CIFAR10", "CIFAR100", "CINIC10L"]:
            x_a = data[:, :, :, 0 : args.half]
            x_b = data[:, :, :, args.half : 32]
        elif args.dataset == "TinyImageNet":
            x_a = data[:, :, :, 0 : args.half]
            x_b = data[:, :, :, args.half : 64]
        elif args.dataset == "Criteo":
            x_b = data[:, args.half : D_]
            x_a = data[:, 0 : args.half]
        elif args.dataset == "BCW":
            x_b = data[:, args.half : 28]
            x_a = data[:, 0 : args.half]
        else:
            raise Exception("Unknown dataset name!")
        if args.test_upper_bound:
            x_b = torch.zeros_like(x_b)
        return x_a, x_b


def swapAttack():
    setting_str = gen_setting_str()
    model = VflFramework(setting_str)
    model = model.cuda()
    cudnn.benchmark = True

    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(
        model.optimizer_top_model, milestones=[stone1, stone2], gamma=args.step_gamma
    )
    lr_scheduler_m_a = torch.optim.lr_scheduler.MultiStepLR(
        model.optimizer_malicious_bottom_model_a,
        milestones=[stone1, stone2],
        gamma=args.step_gamma,
    )
    lr_scheduler_b_b = torch.optim.lr_scheduler.MultiStepLR(
        model.optimizer_benign_bottom_model_b,
        milestones=[stone1, stone2],
        gamma=args.step_gamma,
    )
    schedulers = [lr_scheduler_top_model, lr_scheduler_m_a, lr_scheduler_b_b]

    dir_save_model = args.save_dir + f"/saved_models/{args.dataset}_saved_models"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)
    txt_name = f"{args.dataset}_swapAttack{setting_str}"
    file_path = dir_save_model + "/" + txt_name + ".txt"
    if os.path.exists(file_path):
        os.remove(file_path)


    epoch, attack_period = 0, 0
    val_loader = set_test_loader()

    is_start_swapAttack, grads_a_epochs = False, []
    target_sample_ids, inferred_labels = [], []

    while epoch < args.epochs:
        # shuffle data each epoch.
        train_loader, origin_sample_ids = set_train_loader()

        # judge whether to start swapAttack
        if not is_start_swapAttack:
            if epoch >= args.attack_latest_epoch or (
                judge_converge(grads_a_epochs, args.slope_threshold)
            ):
                is_start_swapAttack = True
                model.gen_labeled_samples(train_loader)

        if not is_start_swapAttack:
            # conduct normal train
            model.simulate_train_epoch(
                train_loader, origin_sample_ids, schedulers, epoch
            )
            model.eval(train_loader, val_loader, origin_sample_ids)
            epoch += 1

            # store average grads
            # TODO: optimize functions here.
            grads_a_average = torch.mean(torch.norm(model.grads_a, dim=1))
            grads_a_epochs.append(grads_a_average.item())
        else:
            # conduct swap attack.
            # for each period of swap attack, we conduct one normal train and n swap train.
            # in the period of swap attack, the train loader keep the same.
            model.gen_target_ids(origin_sample_ids)
            model.simulate_train_epoch(
                train_loader, origin_sample_ids, schedulers, epoch
            )
            model.eval(train_loader, val_loader, origin_sample_ids)
            epoch += 1

            for target_label in list(range(model.num_classes)):
                model.simulate_train_epoch(
                    train_loader,
                    origin_sample_ids,
                    schedulers,
                    epoch,
                    is_swapAttack=True,
                    target_label=target_label,
                )
                epoch += 1

            if (
                len(model.swap_grads_dis[0]) == 0
                or len(model.swap_grads_dis[0][0]) != model.num_classes
            ):
                break

            # gen attack results and fix attack results.
            target_sample_ids_period, inferred_labels_period = [], []
            for batch_id in range(len(model.target_batch_ids)):
                for i, sample_id in enumerate(model.target_sample_ids[batch_id]):
                    diss = np.array(model.swap_grads_dis[batch_id][i])
                    inferred_label = int(diss.argmin())

                    if args.attack_optim:
                        min_abnormal_grad = min(model.swap_grads[batch_id][i])
                        normal_grad = torch.norm(model.grads_a[sample_id], p=2).item()
                        if min_abnormal_grad <= normal_grad * args.optimal_ratio:
                            inferred_label = int(
                                np.array(model.swap_grads[batch_id][i]).argmin()
                            )

                    target_sample_ids_period.append(sample_id)
                    inferred_labels_period.append(inferred_label)

            if len(target_sample_ids_period) > 0:
                target_sample_ids.extend(target_sample_ids_period)
                inferred_labels.extend(inferred_labels_period)

            attack_evaluation(
                train_loader,
                model.num_classes,
                target_sample_ids_period,
                inferred_labels_period,
                attack_period,
                is_attack_finished=False,
                save_to_file=True,
                file_path=file_path,
            )

    attack_evaluation(
        train_loader,
        model.num_classes,
        target_sample_ids,
        inferred_labels,
        is_attack_finished=True,
        save_to_file=True,
        file_path=file_path,
    )
    model.eval(
        train_loader,
        val_loader,
        origin_sample_ids,
        save_to_file=True,
        file_path=file_path,
    )

    # save model.
    torch.save(
        model,
        os.path.join(dir_save_model, f"{args.dataset}_swapAttack{setting_str}.pth"),
        pickle_module=dill,
    )


def attack_evaluation(
    train_loader,
    num_class,
    target_sample_ids,
    inferred_labels,
    attack_period=-1,
    is_attack_finished=False,
    save_to_file=False,
    file_path="",
):
    y_true = np.array(train_loader.dataset.targets)[target_sample_ids].tolist()
    target_samples_count = (
        len(train_loader.dataset.targets) - args.labeled_perclass * num_class
    )

    if len(y_true) != len(inferred_labels):
        raise ValueError("Lengths of y_true and inferred_labels must be the same.")

    correct_count = sum(
        1 for true, pred in zip(y_true, inferred_labels) if true == pred
    )
    attack_accuracy = correct_count / len(y_true)

    if not is_attack_finished:
        print(
            f"{Style.BRIGHT}{Fore.RED}Attack period  : {attack_period}{Style.RESET_ALL}"
        )
    else:
        print(f"{Style.BRIGHT}{Fore.RED}Swap Attack done. {Style.RESET_ALL}")

    print(
        f"{Style.BRIGHT}{Fore.RED}Attack Num     : {len(inferred_labels)}{Style.RESET_ALL}"
    )
    print(
        f"{Style.BRIGHT}{Fore.RED}Attack ratio   : {len(inferred_labels)/target_samples_count: .4f}{Style.RESET_ALL}"
    )
    print(
        f"{Style.BRIGHT}{Fore.RED}Attack accuracy: {attack_accuracy: .4f}{Style.RESET_ALL}"
    )

    if save_to_file:
        savedStdout = sys.stdout
        with open(file_path, "a+") as file:
            sys.stdout = file
            if not is_attack_finished:
                print(f"Attack period  : {attack_period}")
            else:
                print(f"Swap Attack done.")
            print(f"Attack Num     : {len(inferred_labels)}")
            print(f"Attack ratio   : {len(inferred_labels)/target_samples_count: .4f}")
            print(f"Attack accuracy: {attack_accuracy: .4f}\n")
            sys.stdout = savedStdout
        print("Attack evaluation saved to txt!")


def set_train_loader():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)

    origin_sample_ids = list(range(len(train_dataset)))
    random.shuffle(origin_sample_ids)

    if args.dataset == "Criteo":
        train_loader = train_dataset
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=MySequentialSampler(origin_sample_ids)
            # num_workers=args.workers
        )

    # check size_bottom_out and num_classes
    if args.use_top_model is False:
        if dataset_setup.size_bottom_out != dataset_setup.num_classes:
            raise Exception(
                "If no top model is used,"
                " output tensor of the bottom model must equal to number of classes."
            )

    return train_loader, origin_sample_ids


def set_test_loader():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    test_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, False)

    if args.dataset == "Criteo":
        test_loader = test_dataset
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            # num_workers=args.workers
        )

    return test_loader


def gen_setting_str():
    # write experiment setting into file name
    setting_str = ""
    setting_str += "-lr="
    setting_str += str(args.lr)
    if args.dataset != "Yahoo":
        setting_str += "-half="
        setting_str += str(args.half)
    if not args.use_top_model:
        setting_str += "-NoTopModel"
    if not args.swap_attack:
        setting_str += "-NoSwapAttack"
    setting_str += "-epoch="
    setting_str += str(args.epochs)
    if args.swap_attack:
        setting_str += "-attack_latest_epoch="
        setting_str += str(args.attack_latest_epoch)
        setting_str += "-labeled_perclass="
        setting_str += str(args.labeled_perclass)
        setting_str += "-st="
        setting_str += str(args.slope_threshold)
    setting_str += "-batch_size="
    setting_str += str(args.batch_size)
    if args.swap_attack:
        setting_str += "-batch-swap-size="
        setting_str += str(args.batch_swap_size)
    if args.attack_optim:
        setting_str += "-AttackOptimal"
        setting_str += "-optimal_ratio="
        setting_str += str(args.optimal_ratio)
    
    if args.ppdl:
        setting_str += "-ppdl-theta_u="
        setting_str += str(args.ppdl_theta_u)
    if args.gc:
        setting_str += "-gc-preserved_percent="
        setting_str += str(args.gc_preserved_percent)
    if args.lap_noise:
        setting_str += "-lap_noise-scale="
        setting_str += str(args.noise_scale)
    if args.multistep_grad:
        setting_str += "-multistep_grad_bins="
        setting_str += str(args.multistep_grad_bins)
        
    print("settings:", setting_str)
    return setting_str


def set_parser(parser):
    # dataset paras
    parser.add_argument(
        "-d",
        "--dataset",
        default="CIFAR10",
        type=str,
        help="name of dataset",
        choices=[
            "CIFAR10",
            "CIFAR100",
            "TinyImageNet",
            "CINIC10L",
            "Yahoo",
            "Criteo",
            "BCW",
        ],
    )
    parser.add_argument(
        "--path-dataset",
        help="path_dataset",
        type=str,
        default="./data/CIFAR10",
    )
    # framework paras
    parser.add_argument(
        "--use-top-model",
        help="vfl framework has top model or not. If no top model"
        "is used, automatically turn on direct label inference attack,"
        "and report label inference accuracy on the training dataset",
        type=ast.literal_eval,
        default=True,
    )
    parser.add_argument(
        "--test-upper-bound",
        help="if set to True, test the upper bound of our attack: if all the"
        "adversary's samples are labeled, how accurate is the adversary's "
        "label inference ability?",
        type=ast.literal_eval,
        default=False,
    )
    parser.add_argument(
        "--half",
        help="half number of features, generally seen as the adversary's feature num. "
        "You can change this para (lower that party_num) to evaluate the sensitivity "
        "of our attack -- pls make sure that the model to be resumed is "
        "correspondingly trained.",
        type=int,
        default=16,
    )  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    # evaluation & visualization paras
    parser.add_argument("--k", help="top k accuracy", type=int, default=5)
    parser.add_argument(
        "--if-cluster-outputsA",
        help="if_cluster_outputsA",
        type=ast.literal_eval,
        default=False,
    )
    # attack paras
    parser.add_argument(
        "--use-mal-optim",
        help="whether the attacker uses the malicious optimizer",
        type=ast.literal_eval,
        default=False,
    )
    parser.add_argument(
        "--use-mal-optim-all",
        help="whether all participants use the malicious optimizer. If set to "
        "True, use_mal_optim will be automatically set to True.",
        type=ast.literal_eval,
        default=False,
    )
    parser.add_argument(
        "--use-mal-optim-top",
        help="whether the server(top model) uses the malicious optimizer",
        type=ast.literal_eval,
        default=False,
    )
    # saving path paras
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        help="The directory used to save the trained models and csv files",
        default="./saved_experiment_results",
        type=str,
    )
    # possible defenses on/off paras
    parser.add_argument(
        "--ppdl",
        help="turn_on_privacy_preserving_deep_learning",
        type=ast.literal_eval,
        default=False,
    )
    parser.add_argument(
        "--gc",
        help="turn_on_gradient_compression",
        type=ast.literal_eval,
        default=False,
    )
    parser.add_argument(
        "--lap-noise", help="turn_on_lap_noise", type=ast.literal_eval, default=False
    )
    parser.add_argument(
        "--multistep_grad",
        help="turn on multistep-grad",
        type=ast.literal_eval,
        default=False,
    )
    # paras about possible defenses
    parser.add_argument(
        "--ppdl-theta-u",
        help="theta-u parameter for defense privacy-preserving deep learning",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--gc-preserved-percent",
        help="preserved-percent parameter for defense gradient compression",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--noise-scale",
        help="noise-scale parameter for defense noisy gradients",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--multistep_grad_bins",
        help="number of bins in multistep-grad",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--multistep_grad_bound_abs",
        help="bound of multistep-grad",
        type=float,
        default=3e-2,
    )
    # training paras
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of datasets loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )  # TinyImageNet=5e-2, Yahoo=1e-3
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 5e-4)",
    )
    parser.add_argument(
        "--step-gamma",
        default=0.1,
        type=float,
        metavar="S",
        help="gamma for step scheduler",
    )
    parser.add_argument(
        "--stone1", default=50, type=int, metavar="s1", help="stone1 for step scheduler"
    )
    parser.add_argument(
        "--stone2", default=85, type=int, metavar="s2", help="stone2 for step scheduler"
    )
    # swap attack paras
    parser.add_argument(
        "--swap-attack",
        type=ast.literal_eval,
        default=True,
        help="whether to perform a swap attack",
    )
    parser.add_argument(
        "--attack-latest-epoch",
        default=15,
        type=int,
        metavar="N",
        help="the latest time to start the attack",
    )
    parser.add_argument(
        "--labeled-perclass",
        default=1,
        type=int,
        metavar="N",
        help="number of labeled samples in each class or target class.",
    )
    parser.add_argument(
        "--batch-swap-size",
        default=1,
        type=int,
        metavar="N",
        help="Number of samples swapped per batch.",
    )
    parser.add_argument(
        "--slope-threshold",
        "--st",
        default=0.0001,
        type=float,
        metavar="S",
        help="gradient slope threshold to determine whether to start an attack.",
    )
    parser.add_argument(
        "--attack-optim",
        type=ast.literal_eval,
        default=False,
        help="whether to perform attack optimization",
    )
    parser.add_argument(
        "--optimal-ratio",
        default=0.5,
        type=float,
        metavar="S",
        help="Constraints on attack optimization. The smaller it is, the stricter it is..",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The swapping-based label inference attak against vfl framework."
    )
    set_parser(parser)
    args = parser.parse_args()
    swapAttack()
