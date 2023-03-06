import torch


class CategoriesSampler():
    def __init__(self, label, n_batch, n_shot, train_way):
        self.n_batch = n_batch
        self.n_shot = n_shot
        self.label = label
        self.label2id = {}
        for i in range(train_way):
            self.label2id[i] = []
        for idx, all_label in enumerate(label):
            for i, meet in enumerate(all_label):
                if meet == 1:
                    self.label2id[i].append(idx)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i in range(self.n_batch):
            batch = []
            c_idxs = torch.randperm(len(self.label2id))[:5]
            for idx in c_idxs:
                # random.shuffle(self.label2id[l])
                pos = torch.randperm(len(self.label2id[idx.item()]))[:self.n_shot]
                batch.append(torch.LongTensor(self.label2id[idx.item()])[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
