import numpy as np
import torch
from tslearn.metrics import dtw, dtw_path
from utils import unnormalize, normalize
import utils
from loss.dilate_loss import dilate_loss
import properscoring as ps
import time

import train


def eval_base_model(args, model_name, net, loader, norm, gamma, which_split, verbose=1):

    inputs, target, pred_mu, pred_std, pred_d, pred_v = [], [], [], [], [], []

    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    losses_dilate = []
    losses_mse = []
    losses_mae = []
    losses_dtw = []
    losses_tdi = []
    losses_crps = []
    losses_nll = []
    losses_ql = []
    start_time = time.time()

    for i, data in enumerate(loader, 0):
        loss_mse, loss_dtw, loss_tdi, loss_mae, losses_nll, losses_ql = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        # get the inputs
        batch_inputs, batch_target, mask, feats_in, feats_tgt, ids, _ = data
        
        # if args.initialization:
        #     batch_target = utils.get_inputs_median(batch_inputs, batch_target)

        #inputs = torch.tensor(inputs, dtype=torch.float32).to(args.device)
        #batch_target = torch.tensor(batch_target, dtype=torch.float32).to(args.device)
        #feats_in = torch.tensor(feats_in, dtype=torch.float32).to(args.device)
        #feats_tgt = torch.tensor(feats_tgt, dtype=torch.float32).to(args.device)
        #norm = torch.tensor(norm, dtype=torch.float32).to(args.device)
        batch_size, N_output = batch_target.shape[0:2]
        #if N_output == 24:
        #    import ipdb
        #    ipdb.set_trace()
        # DO NOT PASS TARGET during forward pass
        # import pdb ; pdb.set_trace()
        if args.nhead >1:
                mask = mask.transpose(1,0).reshape(-1,args.N_input,args.N_input)
        with torch.no_grad():
            out = net(
                feats_in.to(args.device), batch_inputs.to(args.device), feats_tgt.to(args.device),
                batch_target.to(args.device),mask=mask.to(args.device) if args.mask==1 else None
            )
            if net.is_signature:
                if net.estimate_type in ['point']:
                    batch_pred_mu, _, _ = out
                elif net.estimate_type in ['variance']:
                    batch_pred_mu, batch_pred_d, _, _ = out
                elif net.estimate_type in ['covariance']:
                    batch_pred_mu, batch_pred_d, batch_pred_v, _, _ = out
                elif net.estimate_type in ['bivariate']:
                    batch_pred_mu, batch_pred_d, _, _, _ = out
            else:
                if net.estimate_type in ['point']:
                    batch_pred_mu = out
                elif net.estimate_type in ['variance']:
                    batch_pred_mu, batch_pred_d = out
                elif net.estimate_type in ['covariance']:
                    batch_pred_mu, batch_pred_d, batch_pred_v = out
                elif net.estimate_type in ['bivariate']:
                    batch_pred_mu, batch_pred_d, _ = out
        batch_pred_mu = batch_pred_mu.cpu()
        if net.estimate_type == 'covariance':
            batch_pred_d = batch_pred_d.cpu()
            batch_pred_v = batch_pred_v.cpu()

            #import ipdb; ipdb.set_trace()
            dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                torch.squeeze(batch_pred_mu, dim=-1),
                batch_pred_v,
                torch.squeeze(batch_pred_d, dim=-1)
            )
            batch_pred_std = torch.diagonal(
                dist.covariance_matrix, dim1=-2, dim2=-1).unsqueeze(dim=-1)
            if which_split in ['test']:
                batch_pred_std = norm.unnormalize(batch_pred_std[..., 0], ids=ids, is_var=True).unsqueeze(-1)
        elif net.estimate_type in ['variance', 'bivariate']:
            batch_pred_std = batch_pred_d.cpu()
            batch_pred_v = torch.ones_like(batch_pred_mu) * 1e-9
            if which_split in ['test']:
                batch_pred_std = norm.unnormalize(batch_pred_std[..., 0], ids=ids, is_var=True).unsqueeze(-1)
                batch_pred_d = batch_pred_std**2
        else:
            batch_pred_d = torch.ones_like(batch_pred_mu) * 1e-9
            batch_pred_v = torch.ones_like(batch_pred_mu) * 1e-9
            batch_pred_std = torch.ones_like(batch_pred_mu) * 1e-9

        #batch_target, _ = normalize(batch_target, norm, is_var=False)

        # Unnormalize the data
        if which_split in ['test']:
            batch_pred_mu = norm.unnormalize(batch_pred_mu[..., 0], ids, is_var=False).unsqueeze(-1)

        if which_split in ['test']:
            batch_inputs = norm.unnormalize(batch_inputs[..., 0], ids, is_var=False).unsqueeze(-1)

        inputs.append(batch_inputs)
        target.append(batch_target)
        pred_mu.append(batch_pred_mu)
        pred_std.append(batch_pred_std)
        pred_d.append(batch_pred_d)
        pred_v.append(batch_pred_v)

        del batch_inputs
        del batch_target
        del batch_pred_mu
        del batch_pred_std
        del batch_pred_d
        del batch_pred_v
        #torch.cuda.empty_cache()
        #print(i)

    end_time = time.time()

    inputs = torch.cat(inputs, dim=0)
    target = torch.cat(target, dim=0)
    pred_mu = torch.cat(pred_mu, dim=0)
    pred_std = torch.cat(pred_std, dim=0)
    pred_d = torch.cat(pred_d, dim=0)
    pred_v = torch.cat(pred_v, dim=0)

    # MSE
    #import ipdb ; ipdb.set_trace()
    print('in eval ', target.shape, pred_mu.shape)
    loss_mse = criterion(target, pred_mu).item()
    loss_mae = criterion_mae(target, pred_mu).item()
    loss_smape = 200. * ((torch.abs(target-pred_mu)) / (torch.abs(target) + torch.abs(pred_mu))).mean()

    # DILATE loss
    if model_name in ['seq2seqdilate']:
        loss_dilate, loss_shape, loss_temporal = dilate_loss(target, pred_mu, args.alpha, args.gamma, args.device)
    else:
        loss_dilate = torch.zeros([])
    loss_dilate = loss_dilate.item()

    # DTW and TDI
    loss_dtw, loss_tdi = 0,0
    M = target.shape[0]
    #for k in range(M):
    #    print(k)
    #    target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
    #    output_k_cpu = pred_mu[k,:,0:1].view(-1).detach().cpu().numpy()

    #    loss_dtw += dtw(target_k_cpu,output_k_cpu)
    #    path, sim = dtw_path(target_k_cpu, output_k_cpu)

    #    Dist = 0
    #    for i,j in path:
    #            Dist += (i-j)*(i-j)
    #    loss_tdi += Dist / (N_output*N_output)

    loss_dtw = loss_dtw / M
    loss_tdi = loss_tdi / M

    # CRPS
    loss_crps = ps.crps_gaussian(
        target, mu=pred_mu.detach().numpy(), sig=pred_std.detach().numpy()
    ).mean()

    # CRPS in parts of horizon
    loss_crps_part = []
    N = target.shape[1]
    p = max(int(N/4), 1)
    for i in range(0, N, p):
        if i+p<=N:
            loss_crps_part.append(
                ps.crps_gaussian(
                    target[:, i:i+p],
                    mu=pred_mu[:, i:i+p].detach().numpy(),
                    sig=pred_std[:, i:i+p].detach().numpy()
                ).mean()
            )
    loss_crps_part = np.array(loss_crps_part)

    # NLL
    if net.estimate_type == 'covariance':
        dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
            pred_mu.squeeze(dim=-1), pred_v, pred_d.squeeze(dim=-1)
        )
        #dist = torch.distributions.normal.Normal(pred_mu, pred_std)
        loss_nll = -torch.mean(dist.log_prob(target.squeeze(dim=-1))).item()
        #loss_nll = -torch.mean(dist.log_prob(target)).item()
    elif net.estimate_type in ['variance', 'point', 'bivariate']:
        dist = torch.distributions.normal.Normal(pred_mu, pred_std)
        loss_nll = -torch.mean(dist.log_prob(target)).item()

    quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float)
    #quantiles = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float)
    #quantiles = torch.tensor([0.45, 0.5, 0.55], dtype=torch.float)
    quantile_weights = torch.ones_like(quantiles, dtype=torch.float)
    #quantile_weights = torch.tensor([1., 1., 1.], dtype=torch.float)
    loss_ql = train.QuantileLoss(
        quantiles, quantile_weights
    )(target, pred_mu, pred_std).item()


    metric_dilate = loss_dilate
    metric_mse = loss_mse
    metric_mae = loss_mae
    metric_dtw = loss_dtw
    metric_tdi = loss_tdi
    metric_crps = loss_crps
    metric_crps_part = loss_crps_part
    metric_nll = loss_nll
    metric_ql = loss_ql
    metric_smape = loss_smape.mean()
    total_time = end_time - start_time

    print('Eval dilateloss= ', metric_dilate, \
        'mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi,
        'crps=', metric_crps, 'crps_parts=', metric_crps_part,
        'nll=', metric_nll, 'ql=', metric_ql)

    outputs_dict = dict(
        inputs=inputs, target=target, pred_mu=pred_mu, pred_std=pred_std,
        pred_d=pred_d, pred_v=pred_v
    )
    metrics_dict = dict(
        metric_dilate=metric_dilate, metric_mse=metric_mse,
        metric_dtw=metric_dtw, metric_tdi=metric_tdi,
        metric_crps=metric_crps, metric_mae=metric_mae,
        metric_crps_part=metric_crps_part,
        metric_nll=metric_nll, metric_ql=metric_ql,
        total_time=total_time,
        metric_smape=metric_smape
    )

    return outputs_dict, metrics_dict

def eval_inf_model(args, net, dataset, which_split, gamma, verbose=1):
    '''
        which_split: str (train, dev, test)
    '''
    if which_split in ['train']:
        raise NotImplementedError
    elif which_split in ['dev']:
        loader_str = 'devloader'
        norm_str = 'dev_norm'
    elif which_split in ['test']:
        loader_str = 'testloader'
        norm_str = 'test_norm'

    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()

    num_batches = 0
    for _ in dataset['sum'][1][loader_str]:
        num_batches += 1

    iters = {}
    for agg_method in args.aggregate_methods:
        iters[agg_method] = {}
        for K in args.K_list:
            iters[agg_method][K] = iter(dataset[agg_method][K][loader_str])

    norms = {}
    for agg_method in args.aggregate_methods:
        norms[agg_method] = {}
        for K in args.K_list:
            norms[agg_method][K] = dataset[agg_method][K][norm_str]

    inputs, mapped_ids, target, pred_mu, pred_d, pred_v, pred_std = [], [], [], [], [], [], []
    start_time = time.time()
    for i in range(num_batches):
        dataset_batch = {}
        for agg_method in args.aggregate_methods:
            dataset_batch[agg_method] = {}
            for K in args.K_list:
                dataset_batch[agg_method][K] = iters[agg_method][K].next()

        #import ipdb ; ipdb.set_trace()
        print('Batch id:', i, num_batches)
        batch_pred_mu, batch_pred_d, batch_pred_v, batch_pred_std = net(
            dataset_batch, norms, which_split
        )

        batch_target = dataset_batch['sum'][1][1]

        pred_mu.append(batch_pred_mu.cpu())
        pred_d.append(batch_pred_d.cpu())
        pred_v.append(batch_pred_v.cpu())
        pred_std.append(batch_pred_std.cpu())
        target.append(batch_target.cpu())
        inputs.append(dataset_batch['sum'][1][0])
        mapped_ids.append(dataset_batch['sum'][1][5])

    end_time = time.time()

    pred_mu = torch.cat(pred_mu, dim=0)
    pred_d = torch.cat(pred_d, dim=0)
    pred_v = torch.cat(pred_v, dim=0)
    pred_std = torch.cat(pred_std, dim=0)
    target = torch.cat(target, dim=0)

    inputs = torch.cat(inputs, dim=0)
    mapped_ids = torch.cat(mapped_ids, dim=0)
    inputs = dataset['sum'][1][norm_str].unnormalize(
        inputs[..., 0], ids=mapped_ids
    )
    #import ipdb ; ipdb.set_trace()
    #if which_split in ['dev']:
    #    target = dataset['sum'][1][norm_str].unnormalize(
    #        target[..., 0], ids=mapped_ids
    #    ).unsqueeze(-1)

    # if args.initialization:
    #     target = utils.get_inputs_median(inputs.unsqueeze(-1), target)

    # MSE
    loss_mse = criterion(target, pred_mu)
    loss_mae = criterion_mae(target, pred_mu)
    loss_smape = 200. * ((torch.abs(target-pred_mu)) / (torch.abs(target) + torch.abs(pred_mu))).mean()
    loss_dtw, loss_tdi = 0,0
    # DTW and TDI
    batch_size, N_output = target.shape[0:2]
    for k in range(batch_size):
        target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
        output_k_cpu = pred_mu[k,:,0:1].view(-1).detach().cpu().numpy()

        loss_dtw += dtw(target_k_cpu,output_k_cpu)
        path, sim = dtw_path(target_k_cpu, output_k_cpu)

        Dist = 0
        for i,j in path:
                Dist += (i-j)*(i-j)
        loss_tdi += Dist / (N_output*N_output)

    loss_dtw = loss_dtw /batch_size
    loss_tdi = loss_tdi / batch_size

    # CRPS
    loss_crps = ps.crps_gaussian(
        target, mu=pred_mu.detach().numpy(), sig=pred_std.detach().numpy()
    ).mean()

    #import ipdb ; ipdb.set_trace()
    metric_mse = loss_mse.mean()
    metric_mae = loss_mae.mean()
    metric_dtw = loss_dtw
    metric_tdi = loss_tdi
    metric_crps = loss_crps
    metric_smape = loss_smape.mean()
    total_time = end_time - start_time

    #print('Eval mse= ', metric_mse, ' dtw= ', metric_dtw, ' tdi= ', metric_tdi)
    #import ipdb ; ipdb.set_trace()

    return (
        inputs, target, pred_mu, pred_std, pred_d, pred_v,
        metric_mse, metric_dtw, metric_tdi,
        metric_crps, metric_mae, metric_smape, total_time
    )


def eval_aggregates(inputs, target, mu, std, d, v=None, K_list=None):
    N = target.shape[1]

    criterion = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()

    if K_list is None:
        K_candidates = [1, 2, 3, 4, 6, 12, 24, 30]
    else:
        K_candidates = K_list
    K_list = [K for K in K_candidates if N%K==0]

    agg2metrics = {}
    for agg in ['sum', 'slope', 'diff']:
        agg2metrics[agg] = {}
        for K in K_list:
            agg2metrics[agg][K] = {}
            target_agg = utils.aggregate_data(target[..., 0], agg, K, False).unsqueeze(-1)
            mu_agg = utils.aggregate_data(mu[..., 0], agg, K, False).unsqueeze(-1)
            var_agg = utils.aggregate_data(d[..., 0], agg, K, True, v=v).unsqueeze(-1)
            std_agg = torch.sqrt(var_agg)

            mse = criterion(target_agg, mu_agg).item()
            mae = criterion_mae(target_agg, mu_agg).item()

            crps = ps.crps_gaussian(
                target_agg.detach().numpy(), mu_agg.detach().numpy(),
                std_agg.detach().numpy()
            ).mean()

            agg2metrics[agg][K]['mse'] = mse
            agg2metrics[agg][K]['mae'] = mae
            agg2metrics[agg][K]['crps'] = crps


    return agg2metrics
