import random
from collections import deque

import numpy as np
import torch
from torch import optim

torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, BertModel
from src.utils import modeling, model_utils, model_calib


class bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout):
        super(bert_classifier, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls_hidden = last_hidden[0][:, 0]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


class AdversarialAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AdversarialAttention, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        attention_weights = F.softmax(self.fc2(x), dim=1)
        weighted_input = torch.mul(input, attention_weights)
        return weighted_input


# BERT


# BERTweet
class roberta_large_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout, hidden_size=512, num_layers=1):
        super(roberta_large_classifier, self).__init__()

        self.config = AutoConfig.from_pretrained("C:\\Users\\20171\\Desktop\\Papers &Codes\\Distilling Calibrated "
                                                 "Knowledge for Stance Detection\\CKD-master\\vinai\\bertweet-base",
                                                 local_files_only=True)
        self.roberta = AutoModel.from_pretrained("C:\\Users\\20171\\Desktop\\Papers &Codes\\Distilling Calibrated "
                                                 "Knowledge for Stance Detection\\CKD-master\\vinai\\bertweet-base",
                                                 local_files_only=True)
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.lstm = nn.LSTM(self.config.hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.adversarial_attention = AdversarialAttention(self.config.hidden_size, hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, num_labels)

    def forward(self, **kwargs):
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        cls_hidden = last_hidden[0][:, 0]
        query = self.dropout(cls_hidden)
        # 添加自注意力
        weighted_input = self.adversarial_attention(query)
        lstm_input = weighted_input.unsqueeze(0).repeat(self.num_layers * 2, 1, 1)
        lstm_output, _ = self.lstm(lstm_input)

        # 使用最后一层LSTM的输出进行分类
        out = self.out(lstm_output[-1])

        return out


# born again networks (ban)
class ban_updater(object):
    def __init__(self, **kwargs):
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.s_gen = kwargs.pop("s_gen")
        self.last_model = None
        self.state_size = 2  # 设置状态大小
        self.action_size = 4  # 设置动作大小
        self.agent = DQNAgent(self.state_size, self.action_size)

    def update(self, inputs, criterion, percent, T, args):
        state = self.process_inputs(inputs)  # 将输入转换为状态表示
        action = self.agent.act(state)  # 使用DQN代理选择动作
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        if self.s_gen > 0:
            self.last_model.eval()
            with torch.no_grad():
                teacher_outputs = self.last_model(**inputs).detach()
            loss = self.kd_loss(outputs, inputs['gt_label'], teacher_outputs, percent, T)
        else:
            loss = criterion(outputs, inputs['gt_label'])

        loss.backward()
        if args['clipgradient']:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        next_state = self.process_inputs(inputs)  # 将输入转换为下一个状态
        reward = self.calculate_reward(loss.item())  # 计算奖励
        done = False  # 设置是否结束
        self.agent.remember(state, action, reward, next_state, done)
        self.agent.replay(batch_size=64)  # 进行经验回放

    def process_inputs(self, inputs):
        # 将输入转换为状态表示
        state = []
        # 有两个特征 'input_ids' 和 'attention_mask'
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        state.append(input_ids.float().mean().item())
        state.append(attention_mask.float().mean().item())
        return state

    def calculate_reward(self, loss):
        if loss < 0.001:
            return -0.001
        return -loss

    def register_last_model(self, weight, num_labels, model_select, device, dropout):
        if model_select == 'Bert':
            self.last_model = modeling.bert_classifier(num_labels, model_select, dropout).to(device)
        elif model_select == 'Bertweet':
            self.last_model = modeling.roberta_large_classifier(num_labels, model_select, dropout).to(device)
        self.last_model.load_state_dict(torch.load(weight))

    def get_calib_temp(self, valloader, y_val, device, criterion, dataset):
        with torch.no_grad():
            preds, _ = model_utils.model_preds(valloader, self.last_model, device, criterion)
            T = model_calib.get_best_temp(preds, y_val, dataset)
        return T

    def kd_loss(self, outputs, labels, teacher_outputs, percent, T=1):
        KD_loss = T * T * nn.KLDivLoss(reduction='sum')(F.log_softmax(outputs / T, dim=1),
                                                        F.softmax(teacher_outputs / T, dim=1)) * \
                  (1. - percent) + nn.CrossEntropyLoss(reduction='sum')(outputs, labels) * percent
        return KD_loss


# DQN代理类
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)  # 经验回放缓冲区
        self.gamma = 0.6  # 折扣因子
        self.epsilon = 0.8  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 初始化 Adam 优化器

    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.Tensor(state))
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return  # 如果经验回放缓冲区中的样本数量不足以进行抽样，则直接退出函数

        if batch_size <= 0:
            return  # 如果批次大小是负数或零，则直接退出函数
        minibatch = random.choices(self.memory, k=batch_size) if len(self.memory) >= batch_size else []
        if minibatch:
            for state, action, reward, next_state, done in minibatch:
                state = torch.tensor(state, dtype=torch.float32)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                target = reward
                if not done:
                    next_action_values = self.model(torch.Tensor(next_state))
                    target = (reward + self.gamma * torch.max(next_action_values).item())
                target_f = self.model(torch.Tensor(state)).clone().detach()
                target_f[action] = target
                criterion = nn.SmoothL1Loss(reduction='mean')
                loss = criterion(self.model(torch.Tensor(state)), target_f)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
