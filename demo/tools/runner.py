import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def detect_nan_forward_hook(module, input, output):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def check_tensor(t, name="output"):
        if torch.isnan(t).any():
            print(f"[Forward] ❌ NaN in {name} of {module} on rank {rank}")
        if torch.isinf(t).any():
            print(f"[Forward] ❌ Inf in {name} of {module} on rank {rank}")

    if isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                check_tensor(out, f"output[{i}]")
    elif isinstance(output, torch.Tensor):
        check_tensor(output)
    else:
        print(f"[Forward] ⚠️ Output of {module} is not a Tensor: {type(output)}")

def detect_nan_backward_hook(module, grad_input, grad_output):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def check_tensor(t, name="grad_output"):
        if t is not None:
            if torch.isnan(t).any():
                print(f"[Backward] ❌ NaN in {name} of {module} on rank {rank}")
            if torch.isinf(t).any():
                print(f"[Backward] ❌ Inf in {name} of {module} on rank {rank}")

    for i, g_out in enumerate(grad_output):
        if isinstance(g_out, torch.Tensor):
            check_tensor(g_out, f"grad_output[{i}]")
    for i, g_in in enumerate(grad_input):
        if isinstance(g_in, torch.Tensor):
            check_tensor(g_in, f"grad_input[{i}]")
            
def get_attend_attention_hook(writer):
    def hook(module, input, output):
        if not hasattr(module, 'last_attn'):
            return

        attn = module.last_attn  # shape: [B, H, N, N]
        if not isinstance(attn, torch.Tensor):
            return
        step = getattr(writer, 'step', 0)
        if step % 100 != 0:
            writer.step = step + 1
            return  # ⏩ 跳过这一步1
        
        mean = attn.mean().item()
        std = attn.std().item()
        attn_sample = attn[0, :]
        attn_flat = attn_sample.flatten()
        topk_sum = torch.topk(attn_flat, k=10).values.sum().item()
        total_sum = attn_flat.sum().item()
        topk_ratio = topk_sum / total_sum if total_sum > 0 else 0.0
       
        writer.add_scalar(f'AttendMean/{module.__class__.__name__}', mean, writer.step)
        writer.add_scalar(f'AttendStd/{module.__class__.__name__}', std, writer.step)
        writer.add_scalar(f'AttendTopkRatio/{module.__class__.__name__}', topk_ratio, writer.step)
        

    return hook


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # print model info
    print_log('Trainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)
    
    print_log('Untrainable_parameters:', logger = logger)
    print_log('=' * 25, logger = logger)
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            print_log(name, logger=logger)
    print_log('=' * 25, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
        # base_model._set_static_graph()
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer1 = builder.build_optimizer(base_model, config, stage=1)
    optimizer2 = builder.build_optimizer(base_model, config, stage=2)
    optimizer3 = builder.build_optimizer(base_model, config, stage=3)
    
    # Lightweight sanity check: count parameters per optimizer
    def _count_optim_params(optim):
        try:
            return sum(p.numel() for g in optim.param_groups for p in g.get('params', []))
        except Exception:
            return -1
    print_log(f"[OPTIM INFO] stage1 params: {_count_optim_params(optimizer1)}; "
              f"stage2 params: {_count_optim_params(optimizer2)}; "
              f"stage3 params: {_count_optim_params(optimizer3)}", logger=logger)
    
     # import pdb; pdb.set_trace()
    if args.resume:
        builder.resume_optimizer(optimizer1, optimizer2, optimizer3, args, logger = logger)
        _sched_last_epoch = start_epoch - 1
    else:
        # start_ckpts or cold start: do not try to load optimizer states; start fresh schedule
        _sched_last_epoch = -1
    scheduler1, scheduler2, scheduler3 = builder.build_scheduler(base_model, optimizer1, optimizer2, optimizer3, config, last_epoch=_sched_last_epoch)


    # Criterion
    # ChamferDisL1 = ChamferDistanceL1()
    # ChamferDisL2 = ChamferDistanceL2()
    if isinstance(base_model, torch.nn.parallel.DistributedDataParallel) or isinstance(base_model, torch.nn.DataParallel):
        model_for_hook = base_model.module
    else:
        model_for_hook = base_model
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        train_writer.step = 0
        
    for name, layer in model_for_hook.named_modules():
        # print_log(f"➡️  Visiting layer: {name} ({layer.__class__.__name__})", logger=logger)
        # layer.register_forward_hook(detect_nan_forward_hook)
        # layer.register_full_backward_hook(detect_nan_backward_hook)  
        # import pdb; pdb.set_trace()
        if layer.__class__.__name__ == 'Attend' and rank == 0:
            print_log(f"🧠 Found Attend layer: {name}", logger=logger)

            # 只给 Attend 层注册 attention hook
            layer.register_forward_hook(get_attend_attention_hook(train_writer))
    

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if epoch < config.loss.first_stage:
            consider_metric = config.consider_metric[0]  # only consider chamfer loss in first stage
        elif epoch < config.loss.second_stage:
            consider_metric = config.consider_metric[1]
        else:
            consider_metric = config.consider_metric[2]
        
        if epoch == config.loss.first_stage:
            best_metrics = None  # reset best metrics at the beginning of each stage
        if epoch == config.loss.second_stage:
            best_metrics = None
            
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_losses = AverageMeter(['loss_denoised', 'loss_coarse', 'chamfer_norm1_loss', "total_loss_stage1",
                        'classification_loss', 'mask_loss', 'dice_loss', "total_loss_stage2",
                        'plane_chamfer_loss','plane_normal_loss', 'total_loss_stage3'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ABCPlane' or dataset_name == 'ABCMulti':
                gt = data[0].cuda()        # bs, n, 3
                gt_index = data[1].cuda()  # bs, n
                gt_coeff = data[2].cuda()  # bs, 40, 10
                gt_type = data[3].cuda()   # bs, 40
                pc = gt
                if dataset_name == 'ABCMulti':
                    npoints = config.dataset.train._base_.N_POINTS
                    pc, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                pc = misc.jitter_points(pc, 0.000001)
                pc = pc.cuda()  # bs, n, 3
                if config.dataset.train._base_.augment:
                    pc, gt, gt_coeff, scale = misc.augment_sample_batch_torch(pc, gt, gt_coeff)
                else:
                    scale = None
            elif dataset_name == 'BuildingNL':
                gt = data[0].cuda()        # bs, n, 3
                gt_index = data[1].cuda()  # bs, n
                gt_coeff = data[2].cuda()  # bs, 40, 10
                gt_type = data[3].cuda()   # bs, 40
                pc = data[-1].cuda()  # bs, n, 3
                scale = None
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
            # with torch.autograd.set_detect_anomaly(True):
            ret = base_model(pc, epoch=epoch)
            losses = base_model.module.get_loss(config.loss, ret, gt, gt_index, gt_coeff, gt_type, epoch, scale=scale)
            if epoch < config.loss.first_stage:
                _loss = losses['total_loss_stage1']  
                _loss = _loss / 20
                _loss.backward()
                # forward
                if num_iter == config.step_per_update:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                    num_iter = 0
                    optimizer1.step()
                    base_model.zero_grad()

            elif epoch < config.loss.second_stage:
                _loss = losses['total_loss_stage2']
                _loss = _loss 
                _loss.backward()
                if num_iter == config.step_per_update:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 10), norm_type=2)
                    num_iter = 0
                    optimizer2.step()
                    base_model.zero_grad()  
                
            else:
                # torch.autograd.set_detect_anomaly(True)
                _loss = losses['total_loss_stage3']
                _loss = _loss / 20
                _loss.backward()
                if num_iter == config.step_per_update:
                    # import pdb; pdb.set_trace()
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), getattr(config, 'grad_norm_clip', 1), norm_type=2)
                    num_iter = 0
                    optimizer3.step()
                    base_model.zero_grad()  

            losses_dict = {}
            if args.distributed:
                for key, value in losses.items():
                    losses_dict[key] = dist_utils.reduce_tensor(value, args) * 1000
                train_losses.update(losses_dict)
            else:
                for key, value in losses.items():
                    losses_dict[key] = value * 1000
                train_losses.update(losses_dict)
                
                
            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                for key in losses.keys():
                    train_writer.add_scalar(f'Loss/Batch/{key}', losses[key].item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 10 == 0:
                if epoch < config.loss.first_stage:
                    print_log(f'[Epoch {epoch}/{config.max_epoch}][Batch {idx + 1}/{n_batches}] | BatchTime = {batch_time.val():.3f}s | '
                            f'Losses = [{", ".join(f"{l:.3f}" for l in train_losses.val())}] | lr = {optimizer1.param_groups[0]["lr"]:.6f}', logger=logger)
                elif epoch < config.loss.second_stage:
                    print_log(f'[Epoch {epoch}/{config.max_epoch}][Batch {idx + 1}/{n_batches}] | BatchTime = {batch_time.val():.3f}s | '
                            f'Losses = [{", ".join(f"{l:.3f}" for l in train_losses.val())}] | lr = {optimizer2.param_groups[0]["lr"]:.6f}', logger=logger)
                else:
                    print_log(f'[Epoch {epoch}/{config.max_epoch}][Batch {idx + 1}/{n_batches}] | BatchTime = {batch_time.val():.3f}s | '
                            f'Losses = [{", ".join(f"{l:.3f}" for l in train_losses.val())}] | lr = {optimizer3.param_groups[0]["lr"]:.6f}', logger=logger) 


            if config.scheduler.type == 'GradualWarmup':
                if n_itr < config.scheduler.kwargs_2.total_epoch:
                    if epoch < config.loss.first_stage:
                        scheduler1.step()
                    elif epoch < config.loss.second_stage:
                        scheduler2.step()
                    else:
                        scheduler3.step()

        if isinstance(scheduler1, list):
            if epoch < config.loss.first_stage:
                for item in scheduler1:
                    item.step()
            elif epoch < config.loss.second_stage:
                for item in scheduler2:
                    item.step()
            else:
                for item in scheduler3:
                    item.step()
        else:
            if epoch < config.loss.first_stage:
                scheduler1.step()
            elif epoch < config.loss.second_stage:
                scheduler2.step()
            else:  
                scheduler3.step() 
           
        epoch_end_time = time.time()

        if train_writer is not None:
            for i, key in enumerate(losses.keys()):
                # import pdb; pdb.set_trace()
                train_writer.add_scalar(f'Loss/Epoch/{key}', train_losses.avg(key = key), epoch)
            print_log(f'[Training] Epoch: {epoch} | EpochTime = {epoch_end_time - epoch_start_time:.3f}s | '
                      f'Losses = [{", ".join(f"{l:.4f}" for l in train_losses.avg())}]', logger=logger)

        # Run validation at specified frequency
        if epoch % config.val_freq == 0:
            test_losses = validate(base_model, test_dataloader, epoch, val_writer, config, args, logger=logger)
            if best_metrics is None:
                best_metrics = test_losses[consider_metric]
                metrics = test_losses

            # Save checkpoint if current model is the best so far
            if test_losses[consider_metric] < best_metrics:
                best_metrics = test_losses[consider_metric]
                metrics = test_losses
                builder.save_checkpoint(base_model, optimizer1, optimizer2, optimizer3, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)

        # Save checkpoints
        builder.save_checkpoint(base_model,  optimizer1, optimizer2, optimizer3, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if ((config.loss.first_stage - epoch) < 2  and (config.loss.first_stage - epoch)>=0) or ((config.loss.second_stage - epoch) < 2  and (config.loss.second_stage - epoch) >= 2)or (config.max_epoch - epoch) < 2:
            metrics = test_losses
            builder.save_checkpoint(base_model, optimizer1, optimizer2, optimizer3, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args, logger=logger)
    
    # Close TensorBoard writers
    if train_writer is not None and val_writer is not None:
        train_writer.close()
        val_writer.close()

def validate(base_model, test_dataloader, epoch, val_writer, config, args, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    
    test_losses = AverageMeter(['loss_denoised', 'loss_coarse', 'chamfer_norm1_loss', "total_loss_stage1",
                        'classification_loss', 'mask_loss', 'dice_loss', "total_loss_stage2",
                        'plane_chamfer_loss', 'plane_normal_loss', 'total_loss_stage3'])
    
    n_samples = len(test_dataloader)  # bs is 1

    with torch.no_grad():
        for idx, (model_ids, data) in enumerate(test_dataloader):
            model_id = model_ids[0]
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'ABCPlane' or dataset_name == 'ABCMulti':
                gt = data[0].cuda()        # bs, n, 3
                gt_index = data[1].cuda()  # bs, n
                gt_coeff = data[2].cuda()  # bs, 40, 10
                gt_type = data[3].cuda()   # bs, 40
                pc = gt
                if dataset_name == 'ABCMulti':
                    npoints = config.dataset.train._base_.N_POINTS
                    pc, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                pc = pc.cuda()  # bs, n, 3
                
                if config.dataset.train._base_.augment:
                    pc, gt, gt_coeff, scale = misc.augment_sample_batch_torch(pc, gt, gt_coeff)
                else:
                    scale = None
            elif dataset_name == 'BuildingNL':
                gt = data[0].cuda()        # bs, n, 3
                gt_index = data[1].cuda()  # bs, n
                gt_coeff = data[2].cuda()  # bs, 40, 10
                gt_type = data[3].cuda()   # bs, 40
                pc = data[-1].cuda()  # bs, n, 3
                scale = None
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # Forward pass and loss computation
            ret = base_model(pc, epoch=epoch)
            
            losses = base_model.module.get_loss(config.loss, ret, gt, gt_index, gt_coeff, gt_type, epoch, scale=scale)

            losses_dict = {}
            for key in losses.keys():
                if args.distributed:
                    losses_dict[key] = dist_utils.reduce_tensor(losses[key], args=args) * 1000
                else:
                    losses_dict[key] = losses[key] * 1000
            
            test_losses.update(losses_dict)
            if idx % 10 == 0:
                print_log(f'[Epoch {epoch}/{config.max_epoch}][Batch {idx + 1}/{n_samples}] Losses = [{", ".join(f"{l:.3f}" for l in test_losses.val())}]', logger=logger)
                
        # Synchronize processes in distributed mode
        if args.distributed:
            torch.cuda.synchronize()
    # Log validation metrics to TensorBoard
    if val_writer is not None:
        for i, key in enumerate(losses.keys()):
            val_writer.add_scalar(f'Loss/Epoch/{key}', test_losses.avg(key = key), epoch)

    print_log(f'[Validation] Epoch: {epoch} | Losses = [{", ".join(f"{l:.4f}" for l in test_losses.avg())}]', logger=logger)
    
    # Prepare metrics dictionary for return
    metrics = {}
    for i, key in enumerate(losses.keys()):
        metrics[key] = test_losses.avg(key=key)

    return metrics