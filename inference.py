import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


class Config:
    name = 'fp_exp3'
    model_savename = 'deberta-large'
    if model_savename == 'longformer':
        model_name = '../input/pt-longformer-base'
    elif model_savename == 'roberta-base':
        model_name = '../input/roberta-base'
    elif model_savename == 'roberta-large':
        model_name = '../input/robertalarge'
    elif model_savename == "microsoft/deberta-large":
        model_name = "../input/deberta/large"

    data_dir = '../input/feedback-prize-2021/'
    model_dir = '../input/deberta_model'
    output_dir = '.'

    n_fold = 5
    verbose_steps = 500  # 日志打印频率
    random_seed = 42  # 随机种子

    if model_savename == 'longformer':
        max_length = 1024  # 序列最大长度
        inference_max_length = 4096  # 推理时序列最大长度
        train_batch_size = 4  # train batch size
        valid_batch_size = 4  # valid batch size
        lr = 4e-5
    elif model_savename == 'roberta-base':
        max_length = 512
        inference_max_length = 512
        train_batch_size = 8
        valid_batch_size = 8
        lr = 8e-5
    elif model_savename == 'roberta-large':
        max_length = 512
        inference_max_length = 512
        train_batch_size = 4
        valid_batch_size = 4
        lr = 4e-5
    elif model_savename == "microsoft/deberta-large":
        max_length = 1024  # 序列最大长度
        inference_max_length = 2048  # 推理时序列最大长度
        train_batch_size = 4  # train batch size
        valid_batch_size = 2  # valid batch size
        lr = 4e-5

    num_labels = 15  # labels数量


xm_list = []

IGNORE_INDEX = -100
NON_LABEL = -1
OUTPUT_LABELS = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']
LABELS_TO_IDS = {v: k for k, v in enumerate(OUTPUT_LABELS)}  # LABELS_TO_IDS dict
IDS_TO_LABELS = {k: v for k, v in enumerate(OUTPUT_LABELS)}  # IDS_TO_LABELS dict

# 每种实体的的最小长度阈值，小于阈值不识别
MIN_THRESH = {
    "I-Lead": 11,
    "I-Position": 7,
    "I-Evidence": 12,
    "I-Claim": 1,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

# 每种实体的的最小置信度，小于阈值不识别
PROB_THRESH = {
    "I-Lead": 0.687,
    "I-Position": 0.537,
    "I-Evidence": 0.637,
    "I-Claim": 0.537,
    "I-Concluding Statement": 0.687,
    "I-Counterclaim": 0.37,
    "I-Rebuttal": 0.537,
}

# general
import pandas as pd
import numpy as np
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import gc
from collections import defaultdict
# nlp
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast, AutoConfig, AutoModel, \
    AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# 使用gpu
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')


def agg_essays(train_flg):
    '''
    读取所有txt的内容
    '''
    folder = 'train' if train_flg else 'test'
    names, texts = [], []
    for f in tqdm(list(os.listdir(f'{Config.data_dir}/{folder}'))):
        names.append(f.replace('.txt', ''))  # filename
        texts.append(open(f'{Config.data_dir}/{folder}/' + f, 'r').read())  # text
        df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts


def ner(df_texts, df_train):
    '''
    获取每个token的ner_type(包含B-和I-)
    '''
    all_entities = []
    for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):  # 遍历每个text文件内容
        total = len(row['text_split'])  # 分词长度
        entities = ['O'] * total  #

        for _, row2 in df_train[df_train['id'] == row['id']].iterrows():  # 遍历同一篇文章中的所有discourse
            discourse = row2['discourse_type']  # 获取该discourse的类型
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')]  # predictionstring --> int列表
            entities[list_ix[0]] = f'B-{discourse}'  # 首个token：B-XXX
            for k in list_ix[1:]: entities[k] = f'I-{discourse}'  # 随后的token：I-XXX
        all_entities.append(entities)  # 存入实体列表

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts


def preprocess(df_train=None):
    '''
    text预处理
    '''
    if df_train is None:
        train_flg = False
    else:
        train_flg = True

    df_texts = agg_essays(train_flg)  # 读取txt的内容
    if train_flg:
        df_texts = ner(df_texts, df_train)
    return df_texts


test_texts = preprocess()


class FeedbackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, has_labels):
        self.len = len(dataframe)  # 样本数
        self.data = dataframe  # df数据集
        self.tokenizer = tokenizer  # tokenizer
        self.max_len = max_len  # 最大序列长度
        self.has_labels = has_labels  # 是否有标签

    def __len__(self):
        return self.len  # 样本数

    def __getitem__(self, index):
        text = self.data.text[index]  # 获取text
        encoding = self.tokenizer(
            text.split(),  # 文本内容
            is_split_into_words=True,  # split形式
            padding='max_length',  # 填充至max_len长度
            truncation=True,  # 截断为max_len长度
            max_length=self.max_len
        )

        word_ids = encoding.word_ids()

        # targets
        if self.has_labels:
            word_labels = self.data.entities[index]  # like [B-Lead, I-Lead, I-Lead]
            prev_word_idx = None
            labels_ids = []
            for word_idx in word_ids:
                if word_idx is None:  # 特殊字符
                    labels_ids.append(IGNORE_INDEX)
                else:
                    labels_ids.append(LABELS_TO_IDS[word_labels[word_idx]])

                prev_word_idx = word_idx
            encoding['labels'] = labels_ids
        # convert to torch.tensor
        item = {k: torch.as_tensor(v) for k, v in encoding.items()}
        word_ids2 = [w if w is not None else NON_LABEL for w in word_ids]  # word_ids 为 None，则-1
        item['word_ids'] = torch.as_tensor(word_ids2)

        return item


class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        # 载入 backbone
        if Config.model_savename == 'longformer':
            model_config = LongformerConfig.from_pretrained(Config.model_name)
            self.backbone = LongformerModel.from_pretrained(Config.model_name, config=model_config)
        else:
            model_config = AutoConfig.from_pretrained(Config.model_name)
            self.backbone = AutoModel.from_pretrained(Config.model_name, config=model_config)
        self.model_config = model_config
        self.head = nn.Linear(model_config.hidden_size, Config.num_labels)  # 分类头

    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        logits = self.head(x[0])
        return logits


def build_model_tokenizer():
    if Config.model_savename == 'longformer':
        tokenizer = LongformerTokenizerFast.from_pretrained(Config.model_name, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name, add_prefix_space=True)
    model = FeedbackModel()
    return model, tokenizer


def active_logits(raw_logits, word_ids):
    word_ids = word_ids.view(-1)  # 打平成1维
    active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0],
                                               Config.num_labels)  # 复制成 shape = [word_ids.shape[0], Config.num_labels]
    active_mask = active_mask != NON_LABEL  # token==True，padding和special==False
    active_logits = raw_logits.view(-1, Config.num_labels)
    active_logits = torch.masked_select(active_logits, active_mask)  # return 1dTensor active_logits
    active_logits = active_logits.view(-1, Config.num_labels)
    return active_logits


def active_labels(labels):
    active_mask = labels.view(-1) != IGNORE_INDEX
    active_labels = torch.masked_select(labels.view(-1), active_mask)  # return 1dTensor active_labels
    return active_labels


def active_preds_prob(active_logits):
    active_preds = torch.argmax(active_logits, axis=1)  # argmax
    active_preds_prob, _ = torch.max(active_logits, axis=1)  # max
    return active_preds, active_preds_prob


def inference(model, dl, criterion, valid_flg):
    final_predictions = []
    final_predictions_prob = []
    stream = tqdm(dl)
    model.eval()

    valid_loss = 0
    valid_accuracy = 0
    all_logits = None
    for batch_idx, batch in enumerate(stream, start=1):
        # 将数据喂给GPU
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        with torch.no_grad():
            raw_logits = model(input_ids=ids, mask=mask)  # 模型预测出结果
        del ids, mask

        word_ids = batch['word_ids'].to(device, dtype=torch.long)  # 获取word_ids
        logits = active_logits(raw_logits, word_ids)  # active_logits 去掉mask部分
        sf_logits = torch.softmax(logits, dim=-1)  # logits softmax
        sf_raw_logits = torch.softmax(raw_logits, dim=-1)  # raw_logits softmax
        if valid_flg:  #
            raw_labels = batch['labels'].to(device, dtype=torch.long)  # 获取labels
            labels = active_labels(raw_labels)  # active_labels 去掉mask部分
            preds, preds_prob = active_preds_prob(sf_logits)  # argmax和max
            valid_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())  # 计算accuracy_score
            loss = criterion(logits, labels)  # 计算loss
            valid_loss += loss.item()

        # 存下所有logits结果
        if batch_idx == 1:
            all_logits = sf_raw_logits.cpu().numpy()
        else:
            all_logits = np.append(all_logits, sf_raw_logits.cpu().numpy(), axis=0)

    if valid_flg:
        epoch_loss = valid_loss / batch_idx  # loss
        epoch_accuracy = valid_accuracy / batch_idx  # accuracy
    else:
        epoch_loss, epoch_accuracy = 0, 0
    return all_logits, epoch_loss, epoch_accuracy


def preds_class_prob(all_logits, dl):
    print("predict target class and its probabilty")
    final_predictions = []
    final_predictions_score = []
    stream = tqdm(dl)
    len_sample = all_logits.shape[0]

    for batch_idx, batch in enumerate(stream, start=0):
        for minibatch_idx in range(Config.valid_batch_size):  # batch内部循环
            sample_idx = int(batch_idx * Config.valid_batch_size + minibatch_idx)
            if sample_idx > len_sample - 1: break
            word_ids = batch['word_ids'][minibatch_idx].numpy()  # 某个样本的word_ids
            predictions = []
            predictions_prob = []
            pred_class_id = np.argmax(all_logits[sample_idx], axis=1)  # argmax: class
            pred_score = np.max(all_logits[sample_idx], axis=1)  # max: score
            pred_class_labels = [IDS_TO_LABELS[i] for i in pred_class_id]
            prev_word_idx = -1
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx != prev_word_idx:
                    predictions.append(pred_class_labels[idx])  # argmax: class
                    predictions_prob.append(pred_score[idx])  # max: score
                    prev_word_idx = word_idx
            final_predictions.append(predictions)
            final_predictions_score.append(predictions_prob)
    return final_predictions, final_predictions_score


def get_preds_folds(model, df, dl, criterion, valid_flg=False):
    for i_fold in range(Config.n_fold):
        model_filename = os.path.join(Config.model_dir, f"{Config.model_savename}_{i_fold}.bin")  # 读取模型
        print(f"{model_filename} inference")
        model = model.to(device)  # model to gpu
        model.load_state_dict(torch.load(model_filename))  # 载入权重
        logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg)  # 推理
        if i_fold == 0:
            avg_pred_logits = logits
        else:
            avg_pred_logits += logits
    avg_pred_logits /= Config.n_fold  # n_fold 平均logits
    all_preds, all_preds_prob = preds_class_prob(avg_pred_logits, dl)  # 获取class和score
    df_pred = post_process_pred(df, all_preds, all_preds_prob)  # 后处理
    return df_pred


def post_process_pred(df, all_preds, all_preds_prob):
    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i]  # 文章id
        pred = all_preds[i]  # 某个样本（discourse）的class：like [B-Leader,I-Leader]
        pred_prob = all_preds_prob[i]  # 某个样本（discourse）的score
        j = 0
        while j < len(pred):
            cls = pred[j]  # 某个word的class：like B-Leader
            if cls == 'O':
                j += 1
            else:
                cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls != '':
                avg_score = np.mean(pred_prob[j:end])  # words平均分
                if end - j > MIN_THRESH[cls] and avg_score > PROB_THRESH[cls]:  # 长度和平均分都超过阈值
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(
                        map(str, list(range(j, end))))))  # ['id', 'class', 'new_predictionstring']
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'new_predictionstring']
    return df_pred


model, tokenizer = build_model_tokenizer()  # 构建模型
criterion = nn.CrossEntropyLoss()  # 构建交叉熵损失
ds_test = FeedbackPrizeDataset(test_texts, tokenizer, Config.max_length, False)  # Datasets
dl_test = DataLoader(ds_test, batch_size=Config.train_batch_size, shuffle=False, num_workers=2,
                     pin_memory=True)  # DataLoader
sub = get_preds_folds(model, test_texts, dl_test, criterion)  # 模型融合
sub.columns = ['id', 'class', 'predictionstring']
