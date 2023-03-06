import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from mini_imagenet import PNDataset, my_collate
from samplers import CategoriesSampler
from convnet import PNModel
from utils import pprint, set_gpu, ensure_path, count_acc, euclidean_metric, count_acc, setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    setup_seed(2023)
    #ensure_path(args.save_path)

    trainset = PNDataset('traffic_train')
    train_sampler = CategoriesSampler(trainset.label, 10, args.shot + 2, args.train_way)
    train_loader = DataLoader(dataset=trainset,
                              batch_sampler=train_sampler,
                              collate_fn=my_collate)

    validset = PNDataset('traffic_valid')
    valid_loader = DataLoader(dataset=validset,
                              batch_size=10,
                              collate_fn=my_collate)

    model = PNModel().cuda()

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.5)

    for epoch in range(args.max_epoch):
        lr_scheduler.step()
        model.train()
        loss_all = 0
        tp_all = fp_all = fn_all = tn_all = 0
        for i, batch in enumerate(tqdm(train_loader)):
            sen, label = batch
            sen, label = sen.cuda(), label.cuda()
            p = args.shot * 5
            sen_support, sen_query = sen[:p], sen[p:]
            label_support, label_query = label[:p], label[p:]

            support = model(sen_support)
            query = model(sen_query)

            logits = euclidean_metric(support, query, args.shot,
                                      args.train_way, label_support)
            loss = criterion(torch.abs(logits.float() - 0.001),
                             label_query.float())
            tp, fp, fn, tn = count_acc(logits.float(), label_query.float())
            tp_all += tp
            fp_all += fp
            fn_all += fn
            tn_all += tn
            loss_all += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        precision = tp_all / (tp_all + fp_all + 0.001)
        recall = tp_all / (tp_all + fn_all + 0.001)
        print(
            'epoch {}, loss={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'
            .format(epoch, loss_all, precision, recall,
                    2 * (precision * recall) / (precision + recall + 0.001)))
        
        model.eval()
        loss_all = 0
        tp_all = fp_all = fn_all = tn_all = 0
        for i, batch in enumerate(valid_loader):
            sen, label = batch
            sen, label = sen.cuda(), label.cuda()
            query = model(sen)
            logits = euclidean_metric(support, query, args.shot,
                                      args.train_way, label_support)
            loss = criterion(torch.abs(logits.float() - 0.001),
                             label.float())
            tp, fp, fn, tn = count_acc(logits.float(), label.float())
            tp_all += tp
            fp_all += fp
            fn_all += fn
            tn_all += tn
            loss_all += loss.item()
        precision = tp_all / (tp_all + fp_all + 0.001)
        recall = tp_all / (tp_all + fn_all + 0.001)
        print(
            'epoch {}, loss={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}'
            .format(epoch, loss_all, precision, recall,
                    2 * (precision * recall) / (precision + recall + 0.001)))

