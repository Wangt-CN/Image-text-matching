#encoding:utf-8
import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import seq2vec
import tensorboard_logger as tb_logger
import logging


def train(train_loader, model, criterion, optimizer, epoch, print_freq=10):
    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        images, captions, lengths, ids = train_data
        batch_size = images.size(0)
        margin = 0.2
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_visual = Variable(images)
        input_text = Variable(captions)
        if torch.cuda.is_available():
            input_visual = input_visual.cuda()
            input_text = input_text.cuda()

        #target_answer = Variable(sample['answer'].cuda(async=True))

        # compute output and loss
        scores = model(input_visual, input_text)
        torch.cuda.synchronize()
        loss = utils.calcul_loss(scores, input_visual.size(0), margin)


        train_logger.update('L', loss.cpu().data.numpy())


        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'

                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        train_logger.tb_log(tb_logger, step=model.Eiters)




def validate(val_loader, model, criterion, optimizer, batch_size):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_ii = torch.zeros(5000, 36, 2048)
    input_visual = []
    input_text = []
    ids_ = []

    d = np.zeros((1000, 5000))
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data

        input_ii[ids] = images
        # input_visual.append(images)
        input_text.append(captions)
        ids_.append(ids)

    input_ii = input_ii[[i for i in range(0, 5000, 5)]]
    input_visual = [input_ii[batch_size*i:min(batch_size*(i+1), 1000)] for i in range(1000//batch_size + 1)]
    del input_ii

    for j in range(len(input_visual)):
        for k in range(len(input_text)):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (j, k))

            input_v = input_visual[j]
            input_t = input_text[k]
            batch_size_v = input_v.size(0)
            batch_size_t = input_t.size(0)
            ims = Variable(input_v).cuda()
            txs = Variable(input_t).cuda()
            sums = model(ims, txs)
            # sums = sums.view(batch_size_v, batch_size_t)

            d[batch_size*j:min(batch_size*(j+1), 1000), batch_size*k:min(batch_size*(k+1), 5000)] = sums.cpu().data.numpy()
        sys.stdout.write('\n')
    np.save('stage_1_test', d)

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = r1t + r5t + r10t + r1i + r5i + r10i

    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore

