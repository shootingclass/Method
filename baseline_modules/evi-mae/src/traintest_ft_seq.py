# -*- coding: utf-8 -*-
# Sequence-level fine-tuning train/validate loop for HWU dataset.
# Each batch contains variable-length sequences of windows.
# Windows are encoded individually, then mean-pooled before classification.

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
import wandb
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def train_seq(evi_model, train_loader, test_loader, args):
    """
    Sequence-level fine-tuning.
    
    Each batch from train_loader:
        sensor_padded: (B, S_max, C, H, W)
        video_padded:  (B, S_max, C_v, T_v, H_v, W_v)
        lengths:       (B,)
        labels:        (B, n_class)
        metas:         list of dicts
    
    For each sample, we encode each window with evi_model.forward_embedding,
    then mean-pool over windows, then classify with evi_model.mlp_head.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # Init wandb
    is_master = True
    if 'RANK' in os.environ and int(os.environ['RANK']) > 0:
        is_master = False

    if is_master and args.exp_dir is not None:
        run_name = f"{args.model}_{args.dataset}_ftseq_batch{args.batch_size}_epoch{args.n_epochs}_lr{args.lr}_{args.ftmode}"
        wandb.init(project="Method_Linear_Probe", name=run_name, config=args)

    # evi_model is potentially DataParallel
    if isinstance(evi_model, nn.DataParallel):
        model_to_optim = evi_model.module
    else:
        model_to_optim = evi_model
        evi_model = evi_model.to(device)


    # Separate mlp_head params for higher lr
    mlp_list = ['mlp_head.0.weight', 'mlp_head.0.bias', 'mlp_head.1.weight', 'mlp_head.1.bias',
                'mlp_head_a.0.weight', 'mlp_head_a.0.bias', 'mlp_head_a.1.weight', 'mlp_head_a.1.bias',
                'mlp_head_v.0.weight', 'mlp_head_v.0.bias', 'mlp_head_v.1.weight', 'mlp_head_v.1.bias',
                'mlp_head_concat.0.weight', 'mlp_head_concat.0.bias', 'mlp_head_concat.1.weight', 'mlp_head_concat.1.bias']
    mlp_params = list(filter(lambda kv: kv[0] in mlp_list, model_to_optim.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in mlp_list, model_to_optim.named_parameters()))
    mlp_params = [i[1] for i in mlp_params]
    base_params = [i[1] for i in base_params]

    if args.freeze_base == True:
        print('Pretrained backbone parameters are frozen.')
        for param in base_params:
            param.requires_grad = False

    trainables = [p for p in model_to_optim.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in model_to_optim.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))

    optimizer = torch.optim.Adam(
        [{'params': base_params, 'lr': args.lr * args.base_lr},
         {'params': mlp_params, 'lr': args.lr * args.head_lr}],
        weight_decay=5e-7, betas=(0.95, 0.999)
    )

    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),
            gamma=args.lrscheduler_decay
        )

    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn

    print('now training with {:s}, main metrics: {:s}, loss function: {:s}'.format(
        str(args.dataset), str(main_metrics), str(loss_fn)))

    epoch += 1
    scaler = GradScaler()
    result = np.zeros([args.n_epochs, 4])
    evi_model.train()

    while epoch < args.n_epochs + 1:
        begin_time = time.time()

        if args.only_val == False:
            evi_model.train()
            print('---------------')
            print(datetime.datetime.now())
            print("current #epochs=%s, #steps=%s" % (epoch, global_step))

            for i, (sensor_padded, video_padded, lengths, labels, metas) in enumerate(train_loader):
                # sensor_padded: (B, S_max, C, H, W)
                # video_padded:  (B, S_max, C_v, T_v, H_v, W_v)
                # lengths:       (B,)
                # labels:        (B, n_class)

                B = sensor_padded.shape[0]
                S_max = sensor_padded.shape[1]
                sensor_padded = sensor_padded.to(device, non_blocking=True)
                video_padded = video_padded.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                with autocast():
                    # Optimized forward pass with DataParallel
                    # evi_model handles sequence flattening and pooling internally
                    output = evi_model(sensor_padded, video_padded, args.ftmode, lengths=lengths, chunk_size=args.batch_size)
                    loss = loss_fn(output, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), B)

                if global_step % args.n_print_steps == 0 and global_step != 0:
                    print('Epoch: [{0}][{1}/{2}]\tTrain Loss {loss_meter.val:.4f}\t'.format(
                        epoch, i, len(train_loader), loss_meter=loss_meter), flush=True)
                    if np.isnan(loss_meter.avg):
                        return
                    if is_master:
                        wandb.log({"Train Loss": loss_meter.val, "Epoch": epoch, "Global Step": global_step})

                global_step += 1

        print('start validation')
        stats, valid_loss = validate_seq(evi_model, test_loader, args, phase='Val')

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']
        f1_weighted = stats[0]['f1_weighted']

        print("mAP: {:.6f}".format(mAP))
        print("acc: {:.6f}".format(acc))
        print("f1_weighted: {:.6f}".format(f1_weighted))
        print("AUC: {:.6f}".format(mAUC))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        result[epoch-1, :] = [acc, mAP, mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')

        if is_master:
            wandb.log({
                "Val mAP": mAP, "Val Acc": acc, "Val F1 Weighted": f1_weighted,
                "Val AUC": mAUC, "Val d-prime": d_prime(mAUC), "Val Loss": valid_loss, "Epoch": epoch
            })
        print('validation finished')

        if args.only_val == True:
            exit()

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch
        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if best_epoch == epoch:
            torch.save(evi_model.state_dict(), "%s/models/best_evi_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
        if args.save_model == True:
            if epoch in [50, 100, 150, 199]:
                torch.save(evi_model.state_dict(), "%s/models/evi_model.%d.pth" % (exp_dir, epoch))

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if main_metrics == 'mAP':
                scheduler.step(mAP)
            elif main_metrics == 'acc':
                scheduler.step(acc)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1
        batch_time.reset()
        loss_meter.reset()


def test_seq(evi_model, test_loader, args):
    """Sequence-level testing with wandb logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))

    # Check running rank for DDP
    is_master = True
    if 'RANK' in os.environ and int(os.environ['RANK']) > 0:
        is_master = False
    
    # ensure wandb is available if is_master
    if is_master and wandb.run is None:
        print("Warning: wandb run is not active. Test results will not be logged to wandb.")

    stats, loss = validate_seq(evi_model, test_loader, args, phase='Test')
    mAP = np.mean([stat['AP'] for stat in stats])
    acc = stats[0]['acc']
    f1_weighted = stats[0]['f1_weighted']
    mAUC = np.mean([stat['auc'] for stat in stats])

    print("Test mAP: {:.6f}".format(mAP))
    print("Test acc: {:.6f}".format(acc))
    print("Test f1_weighted: {:.6f}".format(f1_weighted))
    print("Test AUC: {:.6f}".format(mAUC))
    print("Test d_prime: {:.6f}".format(d_prime(mAUC)))
    print("Test loss: {:.6f}".format(loss))

    if is_master and wandb.run is not None:
        wandb.log({
            "Test mAP": mAP,
            "Test Acc": acc,
            "Test F1 Weighted": f1_weighted,
            "Test AUC": mAUC,
            "Test d-prime": d_prime(mAUC),
            "Test Loss": loss
        })
    return stats

def validate_seq(evi_model, val_loader, args, output_pred=False, phase='Val'):
    """Sequence-level validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evi_model = evi_model.to(device)
    evi_model.eval()

    A_predictions, A_targets, A_loss = [], [], []

    with torch.no_grad():
        for i, (sensor_padded, video_padded, lengths, labels, metas) in enumerate(val_loader):
            B = sensor_padded.shape[0]
            S_max = sensor_padded.shape[1]
            sensor_padded = sensor_padded.to(device)
            video_padded = video_padded.to(device)
            lengths = lengths.to(device)

            with autocast():
                output = evi_model(sensor_padded, video_padded, args.ftmode, lengths=lengths, chunk_size=args.batch_size)
                
                # Manual loop removed, using model's forward
                # output is logits (B, n_class)

            predictions = output.to('cpu').detach()
            A_predictions.append(predictions)
            A_targets.append(labels)

            labels_dev = labels.to(device)
            loss = args.loss_fn(output, labels_dev)
            A_loss.append(loss.to('cpu').detach())

    output_all = torch.cat(A_predictions)
    target_all = torch.cat(A_targets)
    loss_avg = np.mean([l.item() for l in A_loss])

    # vpaths argument is required by calculate_stats but not used.
    # Create a dummy list matching the number of samples.
    vpaths = [''] * output_all.shape[0]
    stats = calculate_stats(output_all, target_all, vpaths)

    # Confusion Matrix Logging
    try:
        # Check if wandb is active (it should be imported)
        if wandb.run is not None:
            y_true = np.argmax(target_all.numpy(), axis=1)
            y_pred = np.argmax(output_all.numpy(), axis=1)
            cm = confusion_matrix(y_true, y_pred)
            
            # Use class names from args if available, else indices
            if hasattr(args, 'class_names') and args.class_names:
                class_names = args.class_names
            else:
                class_names = [str(i) for i in range(output_all.shape[1])]

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names
            )
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'{phase} Confusion Matrix')
            
            wandb.log({f"{phase} Confusion Matrix": wandb.Image(fig)})
            plt.close(fig)
    except Exception as e:
        print(f"Failed to log confusion matrix: {e}")

    if output_pred == False:
        return stats, loss_avg
    else:
        return stats, output_all, target_all
