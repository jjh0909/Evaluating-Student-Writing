import os

class Config: 
    name = 'fp_exp3' 

    model_savename = 'microsoft/deberta-large' 
    if model_savename == 'longformer':
        model_name = 'allenai/longformer-base-4096'
    elif model_savename == 'roberta-base':
        model_name = 'roberta-base'
    elif model_savename == 'roberta-large':
        model_name = 'roberta-large'
    elif model_savename == "microsoft/deberta-large":
        model_name = 'microsoft/deberta-large'

    base_dir = 'D:/Python/Feedback Prize/'  # 总路径
    data_dir = base_dir # 数据路径
    model_dir = os.path.join(base_dir, f'model/{name}') # 模型路径
    output_dir = os.path.join(base_dir, f'output/{name}') # 输出路径
    
    n_epoch = 2 
    n_fold = 10
    verbose_steps = 500 # 日志打印频率
    random_seed = 42 # 随机种子

    if model_savename == 'longformer':
        max_length = 1024 # 序列最大长度
        inference_max_length = 4096 # 推理时序列最大长度
        train_batch_size = 4 # train batch size
        valid_batch_size = 4 # valid batch size 
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
        lr = 1e-5
    elif model_savename == "microsoft/deberta-large":
        max_length = 1024 # 序列最大长度
        inference_max_length = 2048 # 推理时序列最大长度
        train_batch_size = 4 # train batch size
        valid_batch_size = 2 # valid batch size 
        lr = 4e-5


    num_labels = 15 # labels数量
    
    is_debug = False 
    if is_debug:
        debug_sample = 1000
        verbose_steps = 16
        n_epoch = 1
        n_fold = 2

xm_list = []

IGNORE_INDEX = -100
NON_LABEL = -1
OUTPUT_LABELS = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
LABELS_TO_IDS = {v:k for k,v in enumerate(OUTPUT_LABELS)} # LABELS_TO_IDS dict
IDS_TO_LABELS = {k:v for k,v in enumerate(OUTPUT_LABELS)} # IDS_TO_LABELS dict

MIN_THRESH = {
    "I-Lead": 11,
    "I-Position": 7,
    "I-Evidence": 12,
    "I-Claim": 1,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

PROB_THRESH = {
    "I-Lead": 0.687,
    "I-Position": 0.537,
    "I-Evidence": 0.637,
    "I-Claim": 0.537,
    "I-Concluding Statement": 0.687,
    "I-Counterclaim": 0.37,
    "I-Rebuttal": 0.537,
}

if not os.path.exists(Config.model_dir):
    get_ipython().system('mkdir $Config.model_dir')

if not os.path.exists(Config.output_dir):
    get_ipython().system('mkdir $Config.output_dir')

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
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast, AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

df_alltrain = pd.read_csv(f'{Config.data_dir}/corrected_train.csv') # 使用修复标签后的train.csv

def agg_essays(train_flg):

    folder = 'train' if train_flg else 'test'
    names, texts =[], []
    for f in tqdm(list(os.listdir(f'{Config.data_dir}/{folder}'))):
        names.append(f.replace('.txt', '')) #filename 
        texts.append(open(f'{Config.data_dir}/{folder}/' + f, 'r').read()) # text 
        df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts

def ner(df_texts, df_train):

    all_entities = []
    for _,  row in tqdm(df_texts.iterrows(), total=len(df_texts)): # 遍历每个text文件内容
        total = len(row['text_split']) # 分词长度
        entities = ['O'] * total #

        for _, row2 in df_train[df_train['id'] == row['id']].iterrows(): # 遍历同一篇文章中的所有discourse
            discourse = row2['discourse_type'] # 获取该discourse的类型
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')] # predictionstring --> int列表
            try:
                entities[list_ix[0]] = f'B-{discourse}' # 首个token：B-XXX
            except:
                continue
            try:
                for k in list_ix[1:]: entities[k] = f'I-{discourse}' # 随后的token：I-XXX
            except:
                continue
        all_entities.append(entities) # 存入实体列表

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts

def preprocess(df_train = None):

    if df_train is None:
        train_flg = False
    else:
        train_flg = True
    
    df_texts = agg_essays(train_flg) # 读取txt的内容
    if train_flg:
        df_texts = ner(df_texts, df_train)
    return df_texts
  
alltrain_texts = preprocess(df_alltrain)
test_texts = preprocess()

if Config.is_debug:
    alltrain_texts = alltrain_texts.sample(Config.debug_sample).reset_index(drop=True)
print(len(alltrain_texts))

def seed_everything(seed=Config.random_seed):

    np.random.seed(seed%(2**32-1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

seed_everything()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

def split_fold(df_train):

    ids = df_train['id'].unique()
    kf = KFold(n_splits=Config.n_fold, shuffle = True, random_state=Config.random_seed)
    for i_fold, (_, valid_index) in enumerate(kf.split(ids)):
        df_train.loc[valid_index,'fold'] = i_fold
    return df_train

alltrain_texts = split_fold(alltrain_texts)
alltrain_texts.head()

class FeedbackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, has_labels):
        self.len = len(dataframe) # 样本数
        self.data = dataframe # df数据集
        self.tokenizer = tokenizer # tokenizer
        self.max_len = max_len # 最大序列长度
        self.has_labels = has_labels # 是否有标签

    def __len__(self):
        return self.len # 样本数
    
    def __getitem__(self, index):
        text = self.data.text[index] # 获取text
        encoding = self.tokenizer(
            text.split(), # 文本内容
            is_split_into_words = True, # split形式
            padding = 'max_length',  # 填充至max_len长度
            truncation = True,  # 截断为max_len长度
            max_length = self.max_len 
        )

        word_ids = encoding.word_ids()  

        # targets
        if self.has_labels:
            word_labels = self.data.entities[index] # like [B-Lead, I-Lead, I-Lead]
            prev_word_idx = None
            labels_ids = []
            for word_idx in word_ids:
                if word_idx is None: # 特殊字符
                    labels_ids.append(IGNORE_INDEX)
                else:
                    labels_ids.append(LABELS_TO_IDS[word_labels[word_idx]])

                prev_word_idx = word_idx
            encoding['labels'] = labels_ids
        # convert to torch.tensor
        item = {k: torch.as_tensor(v) for k, v in encoding.items()}
        word_ids2 = [w if w is not None else NON_LABEL for w in word_ids] # word_ids 为 None，则-1
        item['word_ids'] = torch.as_tensor(word_ids2)

        return item

class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()

        if Config.model_savename == 'longformer':
            model_config = LongformerConfig.from_pretrained(Config.model_name)
            self.backbone = LongformerModel.from_pretrained(Config.model_name, config=model_config)
        else:
            model_config = AutoConfig.from_pretrained(Config.model_name)
            self.backbone = AutoModel.from_pretrained(Config.model_name, config=model_config)
        self.model_config = model_config
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.head = nn.Linear(model_config.hidden_size, Config.num_labels) # 分类头
    
    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
 
        logits1 = self.head(self.dropout1(x[0]))
        logits2 = self.head(self.dropout2(x[0]))
        logits3 = self.head(self.dropout3(x[0]))
        logits4 = self.head(self.dropout4(x[0]))
        logits5 = self.head(self.dropout5(x[0]))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5 # 五层取平均
        return logits

def build_model_tokenizer():

    if Config.model_savename == 'longformer':
        tokenizer = LongformerTokenizerFast.from_pretrained(Config.model_name, add_prefix_space = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name, add_prefix_space = True)
    model = FeedbackModel()

    return model, tokenizer

def active_logits(raw_logits, word_ids):
    word_ids = word_ids.view(-1) # 打平成1维
    active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], Config.num_labels) # 复制成 shape = [word_ids.shape[0], Config.num_labels]
    active_mask = active_mask != NON_LABEL # token==True，padding和special==False
    active_logits = raw_logits.view(-1, Config.num_labels)
    active_logits = torch.masked_select(active_logits, active_mask) # return 1dTensor active_logits
    active_logits = active_logits.view(-1, Config.num_labels) 
    return active_logits

def active_labels(labels):
    active_mask = labels.view(-1) != IGNORE_INDEX
    active_labels = torch.masked_select(labels.view(-1), active_mask) # return 1dTensor active_labels
    return active_labels

def active_preds_prob(active_logits):
    active_preds = torch.argmax(active_logits, axis = 1) # argmax
    active_preds_prob, _ = torch.max(active_logits, axis = 1) # max
    return active_preds, active_preds_prob

def calc_overlap(row):

    set_pred = set(row.new_predictionstring_pred.split(' '))
    set_gt = set(row.new_predictionstring_gt.split(' '))
    # length of each end intersection
    len_pred = len(set_pred)
    len_gt = len(set_gt)
    intersection = len(set_gt.intersection(set_pred))  # 两者交集
    overlap_1 = intersection / len_gt # 交集占ground truth比例
    overlap_2 = intersection / len_pred # 交集占prediction比例
    return [overlap_1, overlap_2]

def score_feedback_comp(pred_df, gt_df):

    gt_df = gt_df[['id', 'discourse_type', 'new_predictionstring']].reset_index(drop = True).copy() # ground truth
    pred_df = pred_df[['id', 'class', 'new_predictionstring']].reset_index(drop = True).copy() # prediction
    gt_df['gt_id'] = gt_df.index
    pred_df['pred_id'] = pred_df.index
    joined = pred_df.merge( 
        gt_df, 
        left_on = ['id', 'class'],
        right_on = ['id', 'discourse_type'], 
        how = 'outer', # outer join
        suffixes = ['_pred', '_gt']  # 重复列的后缀
    )
    joined['new_predictionstring_gt'] =  joined['new_predictionstring_gt'].fillna(' ') # 填充空格
    joined['new_predictionstring_pred'] =  joined['new_predictionstring_pred'].fillna(' ') # 填充空格
    joined['overlaps'] = joined.apply(calc_overlap, axis = 1) # 
    # overlap over 0.5: true positive
    # If nultiple overlaps exists, the higher is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])
 
    # 两种overlap都大于0.5，则为TP
    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis = 1) # 取大的值

    # query: 只取'potential_TP'==True的样本
    # sort_values: 按max_overlap降序
    # groupby: 按['id', 'new_predictionstring_gt'] 分组获取index
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending = False)                  .groupby(['id', 'new_predictionstring_gt']).first()['pred_id'].values

    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids] # FP数据
    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids] # FN数据

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # macro_f1_score 公式
    macro_f1_score = TP / (TP + 1/2 * (FP + FN))
    return macro_f1_score

def oof_score(df_val, oof):

    f1score = []
    classes = ['Lead', 'Position','Claim', 'Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
    # 计算每个class的分数
    for c in classes: 
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val.loc[df_val['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f'{c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    return f1avg

def inference(model, dl, criterion, valid_flg):

    final_predictions = []
    final_predictions_prob = []
    stream = tqdm(dl)
    model.eval()
    
    valid_loss = 0
    valid_accuracy = 0
    all_logits = None
    for batch_idx, batch in enumerate(stream, start = 1):
        # 将数据喂给GPU
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        with torch.no_grad():
            raw_logits = model(input_ids=ids, mask = mask) # 模型预测出结果
        del ids, mask
        
        word_ids = batch['word_ids'].to(device, dtype = torch.long) # 获取word_ids
        logits = active_logits(raw_logits, word_ids) # active_logits 去掉mask部分
        sf_logits = torch.softmax(logits, dim= -1) # logits softmax
        sf_raw_logits = torch.softmax(raw_logits, dim=-1) # raw_logits softmax
        if valid_flg: # 
            raw_labels = batch['labels'].to(device, dtype = torch.long) # 获取labels
            labels = active_labels(raw_labels) # active_labels 去掉mask部分
            preds, preds_prob = active_preds_prob(sf_logits) # argmax和max
            valid_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()) # 计算accuracy_score
            loss = criterion(logits, labels) # 计算loss
            valid_loss += loss.item()
        
        # 存下所有logits结果
        if batch_idx == 1:
            all_logits = sf_raw_logits.cpu().numpy()
        else:
            all_logits = np.append(all_logits, sf_raw_logits.cpu().numpy(), axis=0)

    
    if valid_flg:
        epoch_loss = valid_loss / batch_idx # loss
        epoch_accuracy = valid_accuracy / batch_idx # accuracy
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
        for minibatch_idx in range(Config.valid_batch_size): # batch内部循环
            sample_idx = int(batch_idx * Config.valid_batch_size + minibatch_idx)
            if sample_idx > len_sample - 1 : break
            word_ids = batch['word_ids'][minibatch_idx].numpy() # 某个样本的word_ids
            predictions =[]
            predictions_prob = []
            pred_class_id = np.argmax(all_logits[sample_idx], axis=1) # argmax: class
            pred_score = np.max(all_logits[sample_idx], axis=1) # max: score
            pred_class_labels = [IDS_TO_LABELS[i] for i in pred_class_id] 
            prev_word_idx = -1
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx != prev_word_idx:
                    predictions.append(pred_class_labels[idx]) # argmax: class
                    predictions_prob.append(pred_score[idx]) # max: score
                    prev_word_idx = word_idx 
            final_predictions.append(predictions)
            final_predictions_score.append(predictions_prob)
    return final_predictions, final_predictions_score

def get_preds_onefold(model, df, dl, criterion, valid_flg):
    logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg) # 推理出logits
    all_preds, all_preds_prob = preds_class_prob(logits, dl) # 整理出class和score
    df_pred = post_process_pred(df, all_preds, all_preds_prob) # 后处理
    return df_pred, valid_loss, valid_acc

def post_process_pred(df, all_preds, all_preds_prob):

    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i] # 文章id
        pred = all_preds[i] # 某个样本（discourse）的class：like [B-Leader,I-Leader]
        pred_prob = all_preds_prob[i] # 某个样本（discourse）的score
        j = 0
        while j < len(pred): 
            cls = pred[j] # 某个word的class：like B-Leader
            if cls == 'O': j += 1
            else: cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != 'O' and cls !='':
                avg_score = np.mean(pred_prob[j:end]) # words平均分
                if end - j > MIN_THRESH[cls] and avg_score > PROB_THRESH[cls]: # 长度和平均分都超过阈值
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end)))))) # ['id', 'class', 'new_predictionstring']
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'new_predictionstring']
    return df_pred

def train_fn(model, dl_train, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    train_accuracy = 0
    stream = tqdm(dl_train)
    scaler = GradScaler()

    for batch_idx, batch in enumerate(stream, start = 1):
        # 存入显存
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        raw_labels = batch['labels'].to(device, dtype = torch.long)
        word_ids = batch['word_ids'].to(device, dtype = torch.long)
        optimizer.zero_grad()
        with autocast():
            raw_logits = model(input_ids = ids, mask = mask) # 模型预测
        
        logits = active_logits(raw_logits, word_ids) # get logits
        labels = active_labels(raw_labels) # get labels
        sf_logits = torch.softmax(logits, dim=-1) 
        preds, preds_prob = active_preds_prob(sf_logits) # 获得class和score
        train_accuracy += accuracy_score(labels.cpu().numpy(), preds.cpu().numpy()) # 获取精度
        criterion = nn.CrossEntropyLoss() # 定义交叉熵损失
        loss = criterion(logits, labels) # 计算loss

        scaler.scale(loss).backward() # 反向传播
        scaler.step(optimizer) # 更新优化器
        scaler.update()
        train_loss += loss.item()
        
        # 实时loss打印
        if batch_idx % Config.verbose_steps == 0:
            loss_step = train_loss / batch_idx
            print(f'Training loss after {batch_idx:04d} training steps: {loss_step}')
            
    epoch_loss = train_loss / batch_idx
    epoch_accuracy = train_accuracy / batch_idx
    del dl_train, raw_logits, logits, raw_labels, preds, labels
    torch.cuda.empty_cache()
    gc.collect()
    print(f'epoch {epoch} - training loss: {epoch_loss:.4f}')
    print(f'epoch {epoch} - training accuracy: {epoch_accuracy:.4f}')

def valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion):
    oof, valid_loss, valid_acc  = get_preds_onefold(model, df_val, dl_val, criterion, valid_flg=True) # 获取预测结果
    f1score =[]
    # classes = oof['class'].unique()
    classes = ['Lead', 'Position', 'Claim','Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
    print(f"Validation F1 scores")
    # 计算Evaluate
    for c in classes: 
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val_eval.loc[df_val_eval['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f' * {c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    print(f'Overall Validation avg F1: {f1avg:.4f} val_loss:{valid_loss:.4f} val_accuracy:{valid_acc:.4f}')
    return valid_loss, oof

oof = pd.DataFrame()
for i_fold in range(Config.n_fold):
    print(f'=== fold{i_fold} training ===')
    model, tokenizer = build_model_tokenizer()
    model = model.to(device) # 模型存入gpu
    optimizer = torch.optim.Adam(params=model.parameters(), lr=Config.lr) # 定义Adam优化器
    
    df_train = alltrain_texts[alltrain_texts['fold'] != i_fold].reset_index(drop = True) # train df
    ds_train = FeedbackPrizeDataset(df_train, tokenizer, Config.max_length, True) # train datasets
    df_val = alltrain_texts[alltrain_texts['fold'] == i_fold].reset_index(drop = True) # valid df
    val_idlist = df_val['id'].unique().tolist() 
    
    df_val_eval = df_alltrain.query('id==@val_idlist').reset_index(drop=True)
    ds_val = FeedbackPrizeDataset(df_val, tokenizer, Config.max_length, True) # valid datasets
    dl_train = DataLoader(ds_train, batch_size=Config.train_batch_size, shuffle=True, num_workers=2, pin_memory=True) # train dataloader
    dl_val = DataLoader(ds_val, batch_size=Config.valid_batch_size, shuffle=False, num_workers=2, pin_memory=True) # valid dataloader

    best_val_loss = np.inf
    criterion = nn.CrossEntropyLoss() # 定义交叉熵loss函数

    # epoch训练
    for epoch in range(1, Config.n_epoch + 1):
        train_fn(model, dl_train, optimizer, epoch, criterion) 
        valid_loss, _oof = valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion)
        # 保存模型
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            _oof_fold_best = _oof
            _oof_fold_best['fold'] = i_fold
            model_filename = f'{Config.model_dir}/{Config.model_savename}_{i_fold}.bin'
            torch.save(model.state_dict(), model_filename)
            print(f'{model_filename} saved')

    oof = pd.concat([oof, _oof_fold_best])

print(f'overall cv score: {oof_score(df_train, oof)}')

