import torch
from tqdm import tqdm
from .dataset.asc_dataset import DatasetASC
from .deberta import deberta
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class ASBAModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers_for_pred = [1, 3, 5, 7, 9, 11, 12]
        self.deberta = deberta.DeBERTa(pre_trained='base_mnli')  # Or 'large' or 'base_mnli' or 'large_mnli'
        # Your existing model code
        # do inilization as before
        #
        self.deberta.apply_state()  # Apply the pre-trained model of DeBERTa at the end of the constructor
        self.poolers = nn.ModuleList()
        self.clfs = nn.ModuleList()

        for i in range(7):
            self.poolers.append(Pooler(self.cfg))
            self.clfs.append(nn.Linear(cfg.MODEL.HIDDEN_SIZE, 3))

        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids, input_embedding=None):
        if input_embedding is None:
            encodings, embedding = self.deberta(input_ids, attention_mask, token_type_ids)
        else:
            encodings = self.deberta(None, attention_mask, token_type_ids, input_embedding=input_embedding)

        losses = []
        logits = []
        onehot_labels = F.one_hot(label_ids, num_classes=3).to(dtype=torch.float)

        for count, ind in enumerate(self.layers_for_pred):
            encoding = self.poolers[count](encodings[ind-1])
            logit = self.clfs[count](encoding)
            loss = self.loss(logit, onehot_labels)
            losses.append(loss)
            logits.append(logit)

        losses = torch.sum(torch.stack(losses), dim=0)
        logits = torch.mean(torch.stack(logits), dim=0)

        if input_embedding is None:
            adv_embedding = None
            if self.training:
                losses.backward(retain_graph=True)
                embedding_grad = embedding.grad.detach()
                adv_embedding = embedding + self.cfg.MODEL.EPSILON * embedding_grad / torch.norm(embedding_grad)
            return adv_embedding, losses, logits

        else:
            return losses * self.cfg.MODEL.LAMBDA


class Pooler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.MODEL.HIDDEN_SIZE, cfg.MODEL.HIDDEN_SIZE)

    def forward(self, input):
        pooled = F.tanh(self.linear(input[:, 0]))
        return pooled


class LinearSchedulerWithWarmup():
    def __init__(self, optimizer, max_lr, total_steps, warmup_ratio):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_end_step = int(total_steps * warmup_ratio)
        self.current_step = 1

    def get_lr(self):
        if self.current_step <= self.warmup_end_step:
            lr = self.max_lr * self.current_step / self.warmup_end_step
        else:
            lr = self.max_lr * (self.total_steps - self.current_step) / (self.total_steps - self.warmup_end_step)

        return lr

    def step(self):
        self.optimizer.param_groups[0]["lr"] = self.get_lr()
        self.current_step += 1


def run(cfg):
    train_dataset = DatasetASC(cfg, "train", sources=[cfg.DATA_SOURCE])
    val_dataset = DatasetASC(cfg, "dev", sources=[cfg.DATA_SOURCE])
    test_dataset = DatasetASC(cfg, "test", sources=[cfg.DATA_SOURCE])

    train_loader = DataLoader(train_dataset,
                              sampler=RandomSampler(train_dataset),
                              batch_size=cfg.TRAIN.BATCH,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.TRAIN.BATCH,
                            drop_last=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.TEST.BATCH,
                             drop_last=False)
    model = ASBAModel(cfg)
    model.to(device=cfg.DEVICE)
    for name in [k for (k, v) in model.named_parameters() if (v.requires_grad and "embeddings" not in k)]:
        print(name)
    optimizer = Adam([v for (k, v) in model.named_parameters() if v.requires_grad and "embeddings" not in k],
                     lr=cfg.TRAIN.LR, weight_decay=0.01)
    total_steps = int(len(train_dataset) / cfg.TRAIN.BATCH) * cfg.TRAIN.EPOCH
    scheduler = LinearSchedulerWithWarmup(optimizer,
                                          max_lr=cfg.TRAIN.LR,
                                          total_steps=total_steps,
                                          warmup_ratio=0.1)

    print("Total epochs:{} Dataset length:{} Batch size:{} Total steps:{}".format(cfg.TRAIN.EPOCH,
                                                                                  len(train_dataset),
                                                                                  cfg.TRAIN.BATCH,
                                                                                  total_steps))
    print("Start training")
    for e in range(1, cfg.TRAIN.EPOCH + 1):
        model_loss_epoch = 0
        adv_loss_epoch = 0
        model.train()
        for (input_ids, token_type_ids, attention_mask, label_ids) in tqdm(train_loader):
            input_ids = input_ids.to(device=cfg.DEVICE)
            b = input_ids.shape[0]
            token_type_ids = token_type_ids.to(device=cfg.DEVICE)
            attention_mask = attention_mask.to(device=cfg.DEVICE)
            label_ids = label_ids.to(device=cfg.DEVICE)

            optimizer.zero_grad()
            adv_embedding, loss, _ = model(input_ids, token_type_ids, attention_mask, label_ids, None)
            adv_loss = model(None, token_type_ids, attention_mask, label_ids, adv_embedding)
            adv_loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
            optimizer.step()
            scheduler.step()

            model_loss_epoch += loss.item() * b
            adv_loss_epoch += adv_loss.item() * b

        model_loss_epoch /= len(train_dataset)
        adv_loss_epoch /= len(train_dataset)
        total_loss_epoch = model_loss_epoch + adv_loss_epoch

        print("Epoch:{} model_loss:{} adv_loss:{} total_loss:{}".format(e,
                                                                        model_loss_epoch,
                                                                        adv_loss_epoch,
                                                                        total_loss_epoch))

        correct_count = 0
        val_loss_epoch = 0
        with torch.no_grad():
            model.eval()
            for (input_ids, token_type_ids, attention_mask, label_ids) in val_loader:
                input_ids = input_ids.to(device=cfg.DEVICE)
                b = input_ids.shape[0]
                token_type_ids = token_type_ids.to(device=cfg.DEVICE)
                attention_mask = attention_mask.to(device=cfg.DEVICE)
                label_ids = label_ids.to(device=cfg.DEVICE)
                _, loss, logits = model(input_ids, token_type_ids, attention_mask, label_ids, None)
                onehot_labels = F.one_hot(label_ids, num_classes=3)
                correct_count += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(onehot_labels, dim=1)).item()
                val_loss_epoch += loss * b
            val_loss_epoch /= len(val_dataset)
            accuracy = correct_count / len(val_dataset)
            print("Validation at epoch {}: model_loss:{} accuracy:{}".format(e,
                                                                             val_loss_epoch,
                                                                             accuracy))

    print("Start testing")
    correct_count = 0
    with torch.no_grad():
        model.eval()
        for (input_ids, token_type_ids, attention_mask, label_ids) in test_loader:
            input_ids = input_ids.to(device=cfg.DEVICE)
            b = input_ids.shape[0]
            token_type_ids = token_type_ids.to(device=cfg.DEVICE)
            attention_mask = attention_mask.to(device=cfg.DEVICE)
            label_ids = label_ids.to(device=cfg.DEVICE)
            _, _, logits = model(input_ids, token_type_ids, attention_mask, label_ids, None)
            onehot_labels = F.one_hot(label_ids, num_classes=3)
            correct_count += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(onehot_labels, dim=1)).item()

        accuracy = correct_count / len(test_dataset)
        print("Test result: accuracy:{}".format(accuracy))
