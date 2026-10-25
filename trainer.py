import torch
from torch import nn
import numpy as np
import time
import sys
import SignedAdam
import os
import random
import inspect
from utils import get_poison_tuples
from dataloader import PoisonedDataset, FeatureSet
from experiment_protocol import validate_attack
import torchvision
import torchvision.transforms as transforms

# PLEASE NOTE THAT: the order of the list must be preserved for the consistency of results!
COEFFS = {
    1: [0.4, 0.4, 0.1, 0.1, 0.0],  # Entropy: 1.72
    2: [0.8, 0.05, 0.05, 0.05, 0.05],  # 1.12
    3: [0.9, 0.025, 0.025, 0.025, 0.025],  # 0.67
    4: [0.6, 0.1, 0.1, 0.1, 0.1],  # 1.77
    5: [0.3, 0.3, 0.3, 0.05, 0.05],  # 2
    6: [0.3, 0.2, 0.2, 0.15, 0.15],  # 2.27
    7: [0.22, 0.2, 0.18, 0.17, 0.23],  # 2.31
    8: [0.2, 0.2, 0.2, 0.2, 0.2],  # 2.32
    9: [0.5, 0.3, 0.1, 0.05, 0.05],  # 1.79
    10: [0.5, 0.4, 0.02, 0.02, 0.06],  # 1.47
}


class PoisonBatch(torch.nn.Module):
    """
    Implementing this to work with PyTorch optimizers.
    """

    def __init__(self, base_list):
        super(PoisonBatch, self).__init__()
        base_batch = torch.stack(base_list, 0)
        self.poison = torch.nn.Parameter(base_batch.clone())

    def forward(self):
        return self.poison


def proj_onto_simplex(coeffs, psum=1.0):
    """
    Code stolen from https://github.com/hsnamkoong/robustopt/blob/master/src/simple_projections.py
    Project onto probability simplex by default.
    """
    v_np = coeffs.view(-1).detach().cpu().numpy()
    n_features = v_np.shape[0]
    v_sorted = np.sort(v_np)[::-1]
    cssv = np.cumsum(v_sorted) - psum
    ind = np.arange(n_features) + 1
    cond = v_sorted - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w_ = np.maximum(v_np - theta, 0)
    return torch.Tensor(w_.reshape(coeffs.size())).to(coeffs.device)


def least_squares_simplex(A, b, x_init, tol=1e-6, verbose=False, device='cuda'):
    """
    The inner loop of Algorithm 1
    """
    device = A.device
    m, n = A.size()
    assert b.size()[0] == A.size()[0], 'Matrix and vector do not have compatible dimensions'

    # Initialize the optimization variables
    if x_init is None:
        x = torch.zeros(n, 1).to(device)
    else:
        x = x_init

    # Define the objective function and its gradient
    f = lambda x: torch.norm(A.mm(x) - b).item()
    # change into a faster version when A is a tall matrix
    AtA = A.t().mm(A)
    Atb = A.t().mm(b)
    grad_f = lambda x: AtA.mm(x) - Atb
    # grad_f = lambda x: A.t().mm(A.mm(x)-b)

    # Estimate the spectral radius of the Matrix A'A
    y = torch.normal(0, torch.ones(n, 1)).to(device)
    lipschitz = torch.norm(A.t().mm(A.mm(y))) / torch.norm(y)
    if lipschitz == 0:
        return proj_onto_simplex(x)

    # The stepsize for the problem should be 2/lipschits.  Our estimator might not be correct, it could be too small.  In
    # this case our learning rate will be too big, and so we need to have a backtracking line search to make sure things converge.
    t = 2 / lipschitz

    # Main iteration
    for iter in range(10000):
        x_hat = x - t * grad_f(x)  # Forward step:  Gradient decent on the objective term
        if f(x_hat) > f(x):  # Check whether the learning rate is small enough to decrease objective
            t = t / 2
        else:
            x_new = proj_onto_simplex(x_hat)  # Backward step: Project onto prob simplex
            stopping_condition = torch.norm(x - x_new) / max(torch.norm(x), 1e-8)
            if verbose: print('iter %d: error = %0.4e' % (iter, stopping_condition))
            if stopping_condition < tol:  # check stopping conditions
                break
            x = x_new

    return x


def loss_from_center(subs_net_list, target_feat_list, poison_batch, net_repeat, end2end, feature_mode="original"):
    if feature_mode in ("flat-layerwise", "pool-layerwise"):
        if end2end or net_repeat < 1:
            raise ValueError("Layerwise features require the feature attack representation and net_repeat >= 1")
        losses = []
        for net, centers in zip(subs_net_list, target_feat_list):
            repeats = [extract_attack_features(net, poison_batch(), feature_mode)
                       for _ in range(net_repeat)]
            layer_losses = []
            for layer, center in enumerate(centers):
                features = sum(rep[layer] for rep in repeats) / net_repeat
                diff = features.mean(dim=0, keepdim=True) - center
                denom = torch.norm(center, dim=1)
                if torch.any(denom == 0):
                    raise ValueError("Zero target feature norm in layerwise loss")
                layer_losses.append((torch.norm(diff, dim=1) / denom).mean())
            losses.append(torch.stack(layer_losses).mean())
        return torch.stack(losses).mean()
    if end2end:
        loss = 0
        for net, center_feats in zip(subs_net_list, target_feat_list):
            if net_repeat > 1:
                poisons_feats_repeats = [net(x=poison_batch(), block=True) for _ in range(net_repeat)]
                BLOCK_NUM = len(poisons_feats_repeats[0])
                poisons_feats = []
                for block_idx in range(BLOCK_NUM):
                    poisons_feats.append(
                        sum([poisons_feat_r[block_idx] for poisons_feat_r in poisons_feats_repeats]) / net_repeat)
            elif net_repeat == 1:
                poisons_feats = net(x=poison_batch(), block=True)
            else:
                assert False, "net_repeat set to {}".format(net_repeat)

            net_loss = 0
            for pfeat, cfeat in zip(poisons_feats, center_feats):
                diff = torch.mean(pfeat, dim=0) - cfeat
                diff_norm = torch.norm(diff, dim=1) / torch.norm(cfeat, dim=1)
                net_loss += torch.mean(diff_norm)
            loss += net_loss / len(center_feats)
        loss = loss / len(subs_net_list)

    else:
        loss = 0
        for net, center in zip(subs_net_list, target_feat_list):
            poisons = [
                extract_attack_features(
                    net, poison_batch(), feature_mode
                )
                for _ in range(net_repeat)
            ]
            poisons = sum(poisons) / len(poisons)

            diff = torch.mean(poisons, dim=0) - center
            diff_norm = torch.norm(diff, dim=1) / torch.norm(center, dim=1)
            loss += torch.mean(diff_norm)

        loss = loss / len(subs_net_list)

    return loss


# This is for when the coefficients are fixed, but not to 1/k. This is written originally to address the concerns
# raised by reviewers. We have plan to make it more comprehensive. As of now, it's just for linear transfer learning
# when net_repeat is set to one!
def loss_when_coeffs_fixed(subs_net_list, target_feat_list, poison_batch, coeffs, net_repeat=1, end2end=False):
    assert net_repeat == 1 and end2end == False

    loss = 0
    for net, center in zip(subs_net_list, target_feat_list):
        poisons = net(x=poison_batch(), penu=True)
        diff = torch.sum(coeffs * poisons, dim=0) - center
        diff_norm = torch.norm(diff, dim=1) / torch.norm(center, dim=1)
        loss += torch.mean(diff_norm)

    loss = loss / len(subs_net_list)

    return loss


def get_CP_loss(net_list, target_feature_list, poison_batch, s_coeff_list, net_repeat, tol=1e-6, feature_mode="original"):
    """
    Corresponding to one step of the outer loop (except for updating and clipping) of Algorithm 1
    """
    # assert len(net_list) == 1 or net_repeat == 3
    poison_feat_mat_list = []
    for net in net_list:
        if net_repeat > 1:
            poisons = [extract_attack_features(net, poison_batch(), feature_mode) for _ in range(net_repeat)]
            poisons = sum(poisons) / len(poisons)
        elif net_repeat == 1:
            poisons = extract_attack_features(net, poison_batch(), feature_mode)
        else:
            assert False
        poison_feat_mat_list.append(poisons)

    t = time.time()
    for nn, (pfeat_mat, target_feat) in enumerate(zip(poison_feat_mat_list, target_feature_list)):
        s_coeff_list[nn] = least_squares_simplex(A=pfeat_mat.t().detach(), b=target_feat.t().detach(),
                                                 x_init=s_coeff_list[nn], tol=tol)
    coeffs_time = time.time() - t

    total_loss = 0
    for net, s_coeff, target_feat, poison_feat_mat in zip(net_list, s_coeff_list, target_feature_list,
                                                          poison_feat_mat_list):
        residual = target_feat - torch.sum(s_coeff * poison_feat_mat, 0, keepdim=True)
        target_norm_square = torch.sum(target_feat ** 2)
        recon_loss = 0.5 * torch.sum(residual ** 2) / target_norm_square

        total_loss += recon_loss

    total_loss = total_loss / len(net_list)

    return total_loss, s_coeff_list, coeffs_time


def get_CP_loss_end2end(net_list, target_feature_list, poison_batch, s_coeff_list, net_repeat, tol=1e-6):
    """
    Corresponding to one step of the outer loop (except for updating and clipping) of Algorithm 1
    """
    poison_feat_mat_list = []
    for net in net_list:
        repeats = [net(x=poison_batch(), block=True) for _ in range(net_repeat)]
        poison_feat_mat_list.append([sum(draw[layer] for draw in repeats) / net_repeat
                                    for layer in range(len(repeats[0]))])

    total_loss = 0
    coeffs_time = 0

    for nn, (net, target_feats, poison_feats) in enumerate(zip(net_list, target_feature_list, poison_feat_mat_list)):
        total_loss_tmp = 0
        for n_block, (pfeat, tfeat) in enumerate(zip(poison_feats, target_feats)):
            t = time.time()
            s_coeff_list[nn][n_block] = least_squares_simplex(A=pfeat.view(pfeat.size(0), -1).t().detach(),
                                                              b=tfeat.view(-1, 1).detach(),
                                                              x_init=s_coeff_list[nn][n_block], tol=tol)
            coeffs_time += time.time() - t

            residual = tfeat - torch.sum(s_coeff_list[nn][n_block].unsqueeze(2).unsqueeze(3) * pfeat, 0, keepdim=True)
            target_norm_square = torch.sum(tfeat ** 2)
            recon_loss = 0.5 * torch.sum(residual ** 2) / target_norm_square

            total_loss_tmp += recon_loss
        total_loss += total_loss_tmp / len(poison_feats)

    total_loss = total_loss / len(net_list)

    return total_loss, s_coeff_list, coeffs_time


def make_convex_polytope_poisons(subs_net_list, target_net, base_tensor_list, target, device, opt_method='adam',
                                 lr=0.1, momentum=0.9, iterations=4000, epsilon=0.1,
                                 decay_ites=[10000, 15000], decay_ratio=0.1,
                                 mean=torch.Tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1),
                                 std=torch.Tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1),
                                 chk_path='', poison_idxes=[], poison_label=-1,
                                 tol=1e-6, start_ite=0, poison_init=None, end2end=False, mode='convex',
                                 net_repeat=1, feature_mode="original"):
    validate_attack(mode, feature_mode, end2end, net_repeat, len(base_tensor_list))
    if not subs_net_list:
        raise ValueError("At least one substitute is required")
    if not 0 <= epsilon <= 1 or iterations <= start_ite:
        raise ValueError("Invalid epsilon or iteration range")
    if poison_init is None:
        poison_init = base_tensor_list
    for network in subs_net_list:
        model = network.module if isinstance(network, nn.DataParallel) else network
        if end2end and 'block' not in inspect.signature(model.forward).parameters:
            raise ValueError(type(model).__name__ + ' does not support the original block=True E2E attack')
    for network in list(subs_net_list) + [target_net]:
        for parameter in network.parameters():
            parameter.requires_grad_(False)
    if feature_mode != "original" and mode not in ("mean", "convex"):
        raise ValueError("pool/flat currently support --mode mean or convex only")
    if end2end and feature_mode != "original":
        raise ValueError("MSP feature modes use the feature representation, not the original block objective")
    if feature_mode in ("flat-layerwise", "pool-layerwise") and mode != "mean":
        raise ValueError("Layerwise controlled baseline supports --mode mean only")
    target_net.eval()

    poison_batch = PoisonBatch(poison_init).to(device)

    opt_method = opt_method.lower()
    if opt_method == 'sgd':
        optimizer = torch.optim.SGD(poison_batch.parameters(), lr=lr, momentum=momentum)
    elif opt_method == 'signedadam':
        optimizer = SignedAdam.SignedAdam(poison_batch.parameters(), lr=lr, betas=(momentum, 0.999))
        print("Using Signed Adam")
    elif opt_method == 'adam':
        optimizer = torch.optim.Adam(poison_batch.parameters(), lr=lr, betas=(momentum, 0.999))
    else:
        raise ValueError("Unsupported poison optimizer: " + opt_method)
    # Warm starts keep the schedule appropriate to the requested iteration.
    for milestone in decay_ites:
        if milestone < start_ite:
            for group in optimizer.param_groups:
                group['lr'] *= decay_ratio
    target = target.to(device)
    std, mean = std.to(device), mean.to(device)
    base_tensor_batch = torch.stack(base_tensor_list, 0).to(device)
    base_range01_batch = base_tensor_batch * std + mean

    # Because we have turned on DP for the substitute networks,
    # the target image's feature becomes random.
    # We can try enforcing the convex polytope in one of the multiple realizations of the feature,
    # but empirically one realization is enough.
    target_feat_list = []
    # Coefficients for the convex combination.
    # Initializing from the coefficients of last step gives faster convergence.
    s_init_coeff_list = []
    n_poisons = len(base_tensor_list)
    for n, net in enumerate(subs_net_list):
        net.eval()
        if end2end:
            block_feats = [feat.detach() for feat in net(x=target, block=True)]
            target_feat_list.append(block_feats)
            s_coeff = [torch.ones(n_poisons, 1).to(device) / n_poisons for _ in range(len(block_feats))]
        else:
            features = extract_attack_features(net, target, feature_mode)
            if isinstance(features, list):
                target_feat_list.append([feature.detach() for feature in features])
                print("Layerwise feature dimensions:", [feature.shape[1] for feature in features])
            else:
                target_feat_list.append(features.detach())
            s_coeff = torch.ones(n_poisons, 1).to(device) / n_poisons

        s_init_coeff_list.append(s_coeff)

    # Diagnostics always use the victim's final features. Victims need no block API.
    target_feat_in_target = target_net(x=target, penu=True).detach()
    target_init_coeff = [torch.ones(len(base_tensor_list), 1).to(device) / n_poisons]

    cp_loss_func = get_CP_loss_end2end if end2end else get_CP_loss

    coeffs_time = 0
    poisons_time = 0
    for ite in range(start_ite, iterations):
        if ite in decay_ites:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_ratio
            print("%s Iteration %d, Adjusted lr to %.2e" % (time.strftime("%Y-%m-%d %H:%M:%S"), ite, lr))

        poison_batch.zero_grad()
        t = time.time()
        if mode == 'convex':
            total_loss, s_init_coeff_list, coeffs_time_tmp = cp_loss_func(subs_net_list, target_feat_list, poison_batch,
                                                                          s_init_coeff_list,
                                                                          net_repeat=net_repeat,
                                                                          tol=tol,
                                                                          **({"feature_mode": feature_mode} if not end2end else {}))
        elif mode == 'mean':
            total_loss = loss_from_center(subs_net_list, target_feat_list, poison_batch, net_repeat, end2end, feature_mode=feature_mode)
            coeffs_time_tmp = 0

        elif mode.startswith('coeffs_fixed_type_'):
            coeffs_type = int(mode.split("coeffs_fixed_type_")[1])
            coeffs = COEFFS[coeffs_type]
            random.shuffle(coeffs)
            coeffs = torch.Tensor([coeffs]).t().to(device)
            assert abs(sum(coeffs).item() - 1.0) < 10e-3, print(sum(coeffs).item())
            if ite == start_ite:
                print("coeffs fixed to: {}".format(coeffs))
            total_loss = loss_when_coeffs_fixed(subs_net_list, target_feat_list, poison_batch, coeffs, net_repeat,
                                                end2end)
            coeffs_time_tmp = 0

        coeffs_time += coeffs_time_tmp

        if not torch.isfinite(total_loss):
            raise ValueError("Nonfinite attack loss; inspect target feature norms")
        total_loss.backward()
        optimizer.step()
        poisons_time += time.time() - t

        # clip the perturbations into the range
        perturb_range01 = torch.clamp((poison_batch.poison.data - base_tensor_batch) * std, -epsilon, epsilon)
        perturbed_range01 = torch.clamp(base_range01_batch.data + perturb_range01.data, 0, 1)
        poison_batch.poison.data = (perturbed_range01 - mean) / std

        if ite % 50 == 0 or ite == iterations - 1:
            # whether we are doing convex or mean mode, we want to see the convex loss function for the target victim.
            # Note this unification has done after running the attack for convex method and mean method (0-74), i.e.,
            # for convex 0-99 and mean 0-74 the "loss in target network" is showing different losses for convex vs. mean
            target_loss, target_init_coeff, _ = get_CP_loss([target_net],
                                                             [target_feat_in_target],
                                                             poison_batch,
                                                             target_init_coeff,
                                                             net_repeat=1,
                                                             tol=tol)

            # compute the difference in target
            print(" %s Iteration %d \t Training Loss: %.3e \t Loss in Target Net: %.3e\t  " % (
                time.strftime("%Y-%m-%d %H:%M:%S"), ite, total_loss.item(), target_loss.item()))
            sys.stdout.flush()

            # save the checkpoints
            poison_tuple_list = get_poison_tuples(poison_batch, poison_label)
            torch.save({'poison': poison_tuple_list, 'idx': poison_idxes, 'coeffs_time': coeffs_time,
                        'poisons_time': poisons_time, 'target_loss': target_loss.detach(), 'total_loss': total_loss.detach(),
                        'feature_mode': feature_mode, 'mode': mode, 'end2end': end2end,
                        'attack_representation': 'blocks' if end2end else 'features',
                        'diagnostic_feature': 'victim final penultimate',
                        'epsilon': epsilon, 'iteration': ite,
                        'coeff_list': s_init_coeff_list, 'coeff_list_in_victim': target_init_coeff},
                       os.path.join(chk_path, "poison_%05d.pth" % ite))

    return get_poison_tuples(poison_batch, poison_label)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_network_with_poison(net, target_img, poison_tuple_list, base_idx_list, chk_path, args, save_state=False,
                              return_metrics=False):
    # requires implementing a get_penultimate_params_list() method to get the parameter identifier of the net's last layer
    device = args.device
    model = net.module if isinstance(net, nn.DataParallel) else net
    training_mode = getattr(args, 'victim_training', None) or ('end2end' if getattr(args, 'end2end', False) else 'linear')
    module_mode = getattr(args, 'victim_model_mode', 'eval')
    if training_mode not in ('linear', 'end2end') or module_mode not in ('eval', 'train'):
        raise ValueError("Invalid victim training/module mode")
    if training_mode == 'linear' and module_mode != 'eval':
        raise ValueError("Cached linear features require victim_model_mode=eval")
    full_training = training_mode == 'end2end'
    head_parameters = list(model.get_penultimate_params_list())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_probe = next(((name, parameter) for name, parameter in model.named_parameters()
                           if id(parameter) not in head_ids), None)
    probe_before = backbone_probe[1].detach().clone() if return_metrics and backbone_probe else None
    if args.retrain_epochs < 1:
        raise ValueError("retrain_epochs must be positive")
    if full_training:
        for parameter in net.parameters():
            parameter.requires_grad_(True)
        params = net.parameters()
    else:
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        params = head_parameters
        for parameter in params:
            parameter.requires_grad_(True)

    if args.retrain_opt == 'adam':
        print("Using Adam for retraining")
        optimizer = torch.optim.Adam(params, lr=args.retrain_lr, weight_decay=args.retrain_wd)
    else:
        print("Using SGD for retraining")
        optimizer = torch.optim.SGD(params, lr=args.retrain_lr, momentum=args.retrain_momentum,
                                    weight_decay=args.retrain_wd)

    net.eval()

    criterion = nn.CrossEntropyLoss().to(device)

    # Create the poisoned dataset
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    poisoned_dset = PoisonedDataset(args.train_data_path, subset='others', transform=transform_train,
                                    num_per_label=args.num_per_class, poison_tuple_list=poison_tuple_list,
                                    poison_indices=base_idx_list, subset_group=args.subset_group,
                                    preserve_order=True)

    poisoned_loader = torch.utils.data.DataLoader(poisoned_dset, batch_size=args.retrain_bsize, shuffle=True)

    # The test set of clean CIFAR10
    testset = torchvision.datasets.CIFAR10(root=args.dset_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=500)

    if not full_training:
        # create a dataloader that returns the features
        poisoned_loader = torch.utils.data.DataLoader(FeatureSet(poisoned_loader, net, device=args.device),
                                                      batch_size=args.retrain_bsize, shuffle=True)

    for epoch in range(args.retrain_epochs):
        net.train(module_mode == 'train')
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        time_meter = AverageMeter()

        if epoch in args.lr_decay_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        end_time = time.time()
        for ite, (input, target) in enumerate(poisoned_loader):
            input, target = input.to(device), target.to(device)

            if full_training:
                feat = model.penultimate(input)
            else:
                feat = input
            output = model.linear(feat)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(output, target)[0]

            time_meter.update(time.time() - end_time)
            end_time = time.time()
            loss_meter.update(loss.item(), input.size(0))
            acc_meter.update(prec1.item(), input.size(0))

            if epoch % 30 == 0 and (ite == len(poisoned_loader) - 1):
                print("{2}, Epoch {0}, Iteration {1}, loss {loss.val:.3f} ({loss.avg:.3f}), "
                      "acc {acc.val:.3f} ({acc.avg:.3f})".
                      format(epoch, ite, time.strftime("%Y-%m-%d %H:%M:%S"),
                             loss=loss_meter, acc=acc_meter))
            sys.stdout.flush()

        if epoch == args.retrain_epochs - 1:
            # print the scores for target and base
            net.eval()
            with torch.no_grad():
                target_pred = net(target_img.to(device))
            score, pred = target_pred.topk(1, 1, True, True)
            poison_pred_list = []
            for poison_img, _ in poison_tuple_list:
                base_scores = net(poison_img[None, :, :, :].to(device))
                base_score, base_pred = base_scores.topk(1, 1, True, True)
                poison_pred_list.append(base_pred.item())
            print(
                "Target Label: {}, Poison label: {}, Prediction:{}, Target's Score:{}, Poisons' Predictions:{}".format(
                    args.target_label, args.poison_label, pred[0][0].item(),
                    list(target_pred.detach().view(-1).cpu().numpy()),
                    poison_pred_list))

    # Evaluate the results on the clean test set
    net.eval()
    val_acc_meter = AverageMeter()
    with torch.no_grad():
        for ite, (input, target) in enumerate(test_loader):
            input, target = input.to(device), target.to(device)

            output = net(input)

            prec1 = accuracy(output, target)[0]
            val_acc_meter.update(prec1.item(), input.size(0))

            if False or ite % 100 == 0 or ite == len(test_loader) - 1:
                print("{2} Epoch {0}, Val iteration {1}, "
                      "acc {acc.val:.3f} ({acc.avg:.3f})".
                      format(epoch, ite, time.strftime("%Y-%m-%d %H:%M:%S"), acc=val_acc_meter))

    if save_state:
        state_dict = {"state_dict": net.state_dict(), "epoch": epoch, "acc": val_acc_meter.avg,
                      "victim_training": training_mode, "victim_model_mode": module_mode}
        torch.save(state_dict, os.path.join(chk_path, 'last_epoch.pth'))

    print("* Prec: {}".format(val_acc_meter.avg))
    if return_metrics:
        logits = target_pred.detach().flatten().cpu()
        return dict(prediction=int(pred[0][0]), clean_accuracy_percent=val_acc_meter.avg,
                    victim_training=training_mode, victim_model_mode=module_mode,
                    trainable_parameters=sum(parameter.numel() for parameter in net.parameters() if parameter.requires_grad),
                    backbone_parameter=backbone_probe[0] if backbone_probe else None,
                    backbone_max_change=float((backbone_probe[1].detach()-probe_before).abs().max())
                                        if probe_before is not None else None,
                    logits=logits.tolist(), probabilities=logits.softmax(0).tolist(),
                    train_samples=len(poisoned_dset), poison_count=len(poison_tuple_list))
    return pred[0][0].item()

def extract_attack_features(net, x, feature_mode="original"):
    """Extract differentiable proxy features. DataParallel wraps the model in .module."""
    if feature_mode == "original":
        result = net(x=x, penu=True)
    else:
        model = net.module if isinstance(net, nn.DataParallel) else net
        if feature_mode in ("flat-layerwise", "pool-layerwise"):
            features = model._msp_last_three_features(x)
            if len(features) != 3:
                raise ValueError("Expected exactly three MSP stages")
            if feature_mode == "pool-layerwise":
                result = [torch.nn.functional.avg_pool2d(f, f.shape[-2:]).flatten(1)
                          for f in features]
            else:
                result = [f.flatten(1) for f in features]
            if any(f.ndim != 2 for f in result):
                raise ValueError("Expected each layer to have shape [batch, features]")
            return result
        if feature_mode == "pool":
            result = model.penultimate1(x)
        elif feature_mode == "flat":
            result = model.penultimate2(x)
        else:
            raise ValueError("Unsupported feature mode: {}".format(feature_mode))
    if result.ndim != 2:
        raise ValueError("Expected [batch, features], got {}".format(tuple(result.shape)))
    return result
