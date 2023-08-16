import argparse

from torch import optim
from torch.utils.data.dataloader import DataLoader

from metrics import *
from model import *
from utils import *
import pickle
import torch
import random

import sys
import os
import time

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("seqs_20_20_2l")

parser = argparse.ArgumentParser()

parser.add_argument('--obs_len', type=int, default=20)
parser.add_argument('--pred_len', type=int, default=20)
#parser.add_argument('--dataset', default='eth',
                  #  help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
# parser.add_argument('--clip_grad', type=float, default=None,
#                     help='gadient clipping')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
# parser.add_argument('--momentum', type=float, default=0.9,
#                     help='momentum of lr')
# parser.add_argument('--weight_decay', type=float, default=0.0001,
#                     help='weight_decay on l2 reg')
# parser.add_argument('--lr_sh_rate', type=int, default=100,
#                     help='number of steps to drop the lr')
# parser.add_argument('--milestones', type=int, default=[50, 100],
#                     help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='seq_20_20_2l',
                    help='personal tag for the model ')
parser.add_argument('--gpu_num', default="0", type=str)
parser.add_argument("--seed", type=int, default=72, help="Random seed.")

parser.add_argument('--input_size', default=4, type=int)
parser.add_argument('--hidden_size', default=32, type=int)
parser.add_argument('--output_size', default=4, type=int)
parser.add_argument('--num_layers', default=2, type=int)

parser.add_argument('--embedding_size', default=64, type=int)
parser.add_argument('--dropout', default=0.5, type=int)


args = parser.parse_args()


print("Training initiating....")
print(args)

# 控制台输出记录到文件
class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999, 'min_train_epoch': -1,
                    'min_train_loss': 9999999999999999}

def seq_collate(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)

def train(epoch, model, optimizer, checkpoint_dir, loader_train,loss_function):
    global metrics, constant_metrics
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data 获取数据
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask,seq_start_end= batch

        optimizer.zero_grad()
        pred_pred = model(obs_traj_rel,pred_traj_gt_rel)

        # pred_traj_gt_rel = pred_traj_gt_rel[:, :, :2]

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = loss_function(pred_pred, pred_traj_gt_rel)
            if is_fst_loss:   #train方法最开始设定的值是true
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            # if args.clip_grad is not None:     #默认是10
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count) #每个batch都要打印一次
    metrics['train_loss'].append(loss_batch / batch_count)

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:    #如果train_loss中的最后一个数值 <  最小训练loss
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]    #就将他复制给最小训练loss  -----这就是将最小训练loss保存
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


def vald(epoch, model, checkpoint_dir, loader_val,loss_function):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask,seq_start_end = batch

        with torch.no_grad():
            pred_pred = model(obs_traj_rel,pred_traj_gt_rel)
            # pred_traj_gt_rel = pred_traj_gt_rel[:, :, :2]
            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = loss_function(pred_pred, pred_traj_gt_rel)

                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l

            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                # Metrics
                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):

    random.seed(args.seed)  # 默认72
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  ## 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './dataset/' + args.dataset + '/'

    dset_train = TrajectoryDataset(data_set + 'train/',obs_len=obs_seq_len,pred_len=pred_seq_len,skip=1)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter    每个batch load多少样本  默认是1
        shuffle=True,
        num_workers=0,collate_fn=seq_collate)      #多进程，默认为0  如果出现  brokenerror这个问题的时候，就要检测workers是不是为0

    dset_val = TrajectoryDataset(data_set + 'val/',obs_len=obs_seq_len,pred_len=pred_seq_len,skip=1)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=0,collate_fn=seq_collate)

    print('Training started ...')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    enc = Encoder(input_size=args.input_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                  num_layers=args.num_layers, dropout=args.dropout)
    dec = Decoder( output_size=args.output_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
                  num_layers=args.num_layers, dropout=args.dropout)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TrajSeq(enc,dec,device,teaching_force=0.5).cuda()

    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # if args.use_lrschd:   #默认是true
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    if args.use_lrschd:  # 默认是false
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train,loss_function)    #一次训练之后   一次验证
        vald(epoch, model, checkpoint_dir, loader_val,loss_function)

        writer.add_scalar('trainloss', np.array(metrics['train_loss'])[epoch], epoch)
        writer.add_scalar('valloss', np.array(metrics['val_loss'])[epoch], epoch)

        if args.use_lrschd:     #默认是true
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])    #输出 train_loss   val_loss  这个是输出每个epoch之后的最后一个数值

        print(constant_metrics)   #{'min_val_epoch':  'min_val_loss':   'min_train_epoch':  'min_train_loss':  }
        print('*' * 30)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':

    # 自定义目录存放日志文件
    log_path = 'Logs_train/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    # 记录正常的 print 信息
    sys.stdout = Logger(log_file_name)
    # 记录 traceback 异常信息
    sys.stderr = Logger(log_file_name)

    args = parser.parse_args()
    main(args)
