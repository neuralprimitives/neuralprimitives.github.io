import os, sys
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
from utils.logger import *
from utils.misc import *

def dataset_builder(args, config):
    dataset = build_dataset_from_cfg(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle = shuffle)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.others.bs,
                                            num_workers = int(args.num_workers),
                                            drop_last = config.others.subset == 'train',
                                            worker_init_fn = worker_init_fn,
                                            sampler = sampler,
                                            persistent_workers = True)
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                                shuffle = shuffle, 
                                                drop_last = config.others.subset == 'train',
                                                num_workers = int(args.num_workers),
                                                worker_init_fn=worker_init_fn,
                                                persistent_workers = True)
    return sampler, dataloader

def model_builder(config):
    model = build_model_from_cfg(config)
    return model

def build_optimizer(base_model, config, stage=None):
    opti_config = config.optimizer
    
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=(), stage=None):
            decay = []
            no_decay = []
            parameters = []
            if stage is not None:
                if stage == 1:
                    parameters = [(name, param) for name, param in model.named_parameters() if ('primitive_segmentation' not in name and 'plane_segmentation' not in name)]
                elif stage == 2:
                    parameters = [(name, param) for name, param in model.named_parameters() if ('primitive_segmentation' in name or 'plane_segmentation' in name)]  # only optimize segmentation head
                else:
                    parameters = [(name, param) for name, param in model.named_parameters()]
            else:
                parameters = [(name, param) for name, param in model.named_parameters()]
                
            for name, param in parameters:
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]

        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay, stage=stage)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    else:
        if stage is not None:
            if stage == 1:
               parameters = [param for name, param in base_model.named_parameters() if ('primitive_segmentation' not in name and 'plane_segmentation' not in name)]
            elif stage == 2:
                parameters = [param for name, param in base_model.named_parameters() if ('primitive_segmentation' in name or 'plane_segmentation' in name)]
            else:
                parameters = base_model.parameters()
        else:            
            parameters = base_model.parameters()
        if opti_config.type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, parameters), **opti_config.kwargs)
        elif opti_config.type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, parameters), **opti_config.kwargs)
        else:
            raise NotImplementedError()

    return optimizer

def build_scheduler(base_model, optimizer1, optimizer2, optimizer3 , config, last_epoch=-1):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        if last_epoch == -1:
            scheduler1 = build_lambda_sche(optimizer1, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
            scheduler2 = build_lambda_sche(optimizer2, sche_config.kwargs, last_epoch=last_epoch) 
            scheduler3 = build_lambda_sche(optimizer3, sche_config.kwargs, last_epoch=last_epoch)
        else:
            scheduler1 = build_lambda_sche(optimizer1, sche_config.kwargs, last_epoch=last_epoch)
            scheduler2 = build_lambda_sche(optimizer2, sche_config.kwargs, last_epoch=max(-1, last_epoch - config.loss.first_stage))
            scheduler3 = build_lambda_sche(optimizer3, sche_config.kwargs, last_epoch=max(-1, last_epoch - config.loss.second_stage))
        
    elif sche_config.type == 'StepLR':
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, last_epoch=last_epoch, **sche_config.kwargs)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, last_epoch=max(-1, last_epoch - config.loss.first_stage), **sche_config.kwargs)
        scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, last_epoch=max(-1, last_epoch - config.loss.second_stage), **sche_config.kwargs)
       
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr1 = torch.optim.lr_scheduler.StepLR(optimizer1, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler_steplr2 = torch.optim.lr_scheduler.StepLR(optimizer2, last_epoch=max(-1, last_epoch - config.loss.first_stage), **sche_config.kwargs_1)
        scheduler_steplr3 = torch.optim.lr_scheduler.StepLR(optimizer3, last_epoch=max(-1, last_epoch - config.loss.second_stage), **sche_config.kwargs_1)
        scheduler1 = GradualWarmupScheduler(optimizer1, after_scheduler=scheduler_steplr1, **sche_config.kwargs_2)
        scheduler2 = GradualWarmupScheduler(optimizer2, after_scheduler=scheduler_steplr2, **sche_config.kwargs_2)
        scheduler3 = GradualWarmupScheduler(optimizer3, after_scheduler=scheduler_steplr3, **sche_config.kwargs_2)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler1 = [scheduler1, bnscheduler]
        scheduler2 = [scheduler2, bnscheduler]
        scheduler3 = [scheduler3, bnscheduler]
    
    return scheduler1, scheduler2, scheduler3

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)

    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer1, optimizer2, optimizer3, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return False, False, False
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    loaded_flags = [False, False, False]
    for idx, opt in enumerate([optimizer1, optimizer2, optimizer3], start=1):
        key = f'optimizer{idx}'
        if key not in state_dict:
            print_log(f"[RESUME WARN] '{key}' not found in checkpoint. Using fresh state.", logger=logger)
            continue
        try:
            opt.load_state_dict(state_dict[key])
            loaded_flags[idx - 1] = True
        except (ValueError, RuntimeError) as e:
            print_log(f"[RESUME WARN] Skipping load for {key} due to incompatibility: {e}", logger=logger)
            continue
    return tuple(loaded_flags)

def save_checkpoint(base_model, optimizer1, optimizer2, optimizer3, epoch, metrics, best_metrics, prefix, args, logger = None):
    if args.local_rank == 0:
        torch.save({
                    'base_model' : base_model.module.state_dict() if args.distributed else base_model.state_dict(),
                    'optimizer1' : optimizer1.state_dict(),
                    'optimizer2' : optimizer2.state_dict(),
                    'optimizer3' : optimizer3.state_dict(),
                    'epoch' : epoch,
                    'metrics' : metrics if metrics is not None else dict(),
                    'best_metrics' :best_metrics if best_metrics is not None else dict(),
                    }, os.path.join(args.experiment_path, prefix + '.pth'))
        print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return epoch