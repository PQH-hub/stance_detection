import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import json
import os
from utils import preprocessing as pp
import utils.data_helper as dahe
from transformers import AdamW
from src.utils import modeling, evaluation, model_utils, model_calib

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def run_classifier():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', help='Name of the config data file', default='../config/config-bert.txt',
                        required=False)
    parser.add_argument('-train', '--train_data', help='Name of the training data file',
                        default='../data/covid19/raw_train_all_onecol.csv', required=False)
    parser.add_argument('-dev', '--dev_data', help='Name of the dev data file',
                        default='../data/covid19/raw_val_all_onecol.csv', required=False)
    parser.add_argument('-test', '--test_data', help='Name of the test data file',
                        default='../data/covid19/raw_test_all_onecol.csv', required=False)
    parser.add_argument('-outdir', '--model_dir', help='Dir of the saved model',
                        default='C:/Users/20171/Desktop/Papers &Codes/Distilling Calibrated Knowledge for Stance '
                                'Detection/CKD-master/model',
                        required=False)
    parser.add_argument('-dataset', '--dataset', help='Dataset name', default="covid19", required=False)
    parser.add_argument('-step', '--savestep', type=int, default=5, required=False)
    parser.add_argument('-s', '--seed', type=int, help='Random seed', required=False)
    parser.add_argument('-d', '--dropout', type=float, default=0., help='Dropout rate', required=False)
    parser.add_argument('-lr1', '--lr1', type=float, default=2e-5, help='lr for the main model', required=False)
    parser.add_argument('-lr2', '--lr2', type=float, default=1e-3, help='lr for the output layer', required=False)
    parser.add_argument('-gen', '--n_gen', type=int, default=4, help='Number of generations', required=False)
    parser.add_argument('-l', '--lambd', type=float, help='Weight of KD and CE losses', default=0., required=False)
    parser.add_argument('-t', '--temp', type=float, help='Fixed temperature in KD', default=1, required=False)
    parser.add_argument('-clipgrad', '--clipgradient', help='Clip gradient when over 1', action='store_true',
                        default=True)
    parser.add_argument('-anneal', '--teacher_anneal', action='store_true', default=True)
    parser.add_argument('-calib', '--calib_kd', action='store_true', default=True)

    args = vars(parser.parse_args())

    # 加载config.text文件
    with open(args['config_file'], 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    # 将config文件内的参数赋值给args并打印log方便观察
    random_seeds = [3, 1, 2, 0, args['seed']]
    config['bert_lr'] = args['lr1']
    config['fc_lr'] = args['lr2']
    dataset = args['dataset']
    lambd = args['lambd']
    T = args['temp']
    model_select = config['model_select']
    outdir = args['model_dir']
    s_gen = 0
    n_gen = args['n_gen']
    dropout = args['dropout']
    print("Dataset: ", args['dataset'])
    print("PLM lr: ", config['bert_lr'])
    print("FC lr: ", config['fc_lr'])
    print("Batch size: ", config['batch_size'])
    print("Dropout: ", args['dropout'])
    print("Clip gradient: ", args['clipgradient'])
    print("Lambda used to weigh the importance of KD and CE losses: ", args['lambd'])
    print("Initial or fixed temperature: ", args['temp'])
    print("Number of generations: ", n_gen)
    print("Teacher annealing: ", args['teacher_anneal'])
    print("Calibrated knowledge distillation: ", args['calib_kd'])
    print(60 * "#")

    # 为进程创建归一化字典
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    norm_dict = {**data1, **data2}

    # 使用GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    # load train/val/test sets
    x_train, y_train, x_train_target = pp.clean_all(args['train_data'], dataset, norm_dict)
    x_val, y_val, x_val_target = pp.clean_all(args['dev_data'], dataset, norm_dict)
    x_test, y_test, x_test_target = pp.clean_all(args['test_data'], dataset, norm_dict)

    print(60 * "#")
    print("Size of train set:", len(x_train))
    print("Size of val set:", len(x_val))
    print("Size of test set:", len(x_test))

    num_labels = len(set(y_train))
    x_train_all = [x_train, y_train, x_train_target]
    x_val_all = [x_val, y_val, x_val_target]
    x_test_all = [x_test, y_test, x_test_target]

    best_result, best_val = [], []
    for seed in random_seeds:
        print("current random seed: ", seed)

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 数据加载
        loader, gt_label = dahe.data_helper_bert(x_train_all, x_val_all, x_test_all, model_select, config)
        trainloader, valloader, testloader, trainloader2 = loader[0], loader[1], loader[2], loader[3]
        y_train, y_val, y_test, y_train2 = gt_label[0], gt_label[1], gt_label[2], gt_label[3]
        y_val, y_test, y_train2 = y_val.to(device), y_test.to(device), y_train2.to(device)

        # 训练步骤
        model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, dropout)
        print("Train setup has finished!")
        loss_function = nn.CrossEntropyLoss(reduction='sum')
        loss_function2 = nn.KLDivLoss(reduction='sum')
        kwargs = {
            "model": model,
            "optimizer": optimizer,
            "s_gen": s_gen,
            "n_gen": n_gen,
        }
        updater = modeling.ban_updater(**kwargs)
        best_val, best_val_loss = 0, 10000
        best_val_list, best_test_list = [], []
        best_test_micro, best_test_macro = [], []
        # 开始训练
        diff_seed = [11, 12, 13, 14, 15]
        for gen in range(s_gen, n_gen):
            step = 0
            num_train_steps = int(len(y_train) / int(config['batch_size']) * int(config['total_epochs']))
            print("Total number of training steps: ", num_train_steps)
            for epoch in range(0, int(config['total_epochs'])):
                print(60 * "#")
                print('Epoch {} of gen {}:'.format(epoch, gen))
                # 训练,目的是计算和更新模型的损失值，以便进行梯度下降优化
                updater.model.train()
                for b_id, sample_batch in enumerate(trainloader):
                    optimizer.zero_grad()
                    dict_batch = model_utils.batch_fn(sample_batch)

                    inputs = {k: v.to(device) for k, v in dict_batch.items()}
                    state = updater.process_inputs(inputs)  # 将输入转换为状态表示
                    action = updater.agent.act(state)  # 使用DQN代理选择动作
                    step += 1
                    if args['teacher_anneal']:
                        percent = step / num_train_steps
                        percent = percent if percent <= 1.0 else 1.0
                    else:
                        percent = 1 - float(args['lambd'])

                    loss = updater.update(inputs, loss_function, percent, T, args)
                    # evaluation on dev set
                    split_step = len(trainloader) // args['savestep']
                    if step % split_step == 0:
                        updater.model.eval()
                        with torch.no_grad():
                            preds, loss_val = model_utils.model_preds(valloader, updater.model, device, loss_function)
                            avg_val_loss = sum(loss_val) / len(y_val)
                            f1_average = evaluation.compute_f1(preds, y_val, dataset)
                            preds, _ = model_utils.model_preds(testloader, updater.model, device, loss_function)

                        # save model weights
                        if best_val_loss > avg_val_loss:
                            best_val_loss = avg_val_loss
                            best_val = f1_average
                            last_model_weight = os.path.join(outdir,
                                                             model_select + '_' + str(gen) + '_seed{}.pt'.format(seed))
                            torch.save(updater.model.state_dict(), last_model_weight)
                            print("Best validation result is updated at epoch {}, as: {}".format(epoch, best_val))
                        updater.model.train()
                        # 将输入转换为下一个状态
                        next_inputs = {k: v.to(device) for k, v in dict_batch.items()}
                        next_state = updater.process_inputs(next_inputs)
                        # 根据损失计算奖励值
                        reward = updater.calculate_reward(avg_val_loss)
                        # 将状态、动作、奖励和下一个状态添加到经验回放缓冲区中
                        done = False  # 设置是否结束
                        updater.agent.remember(state, action, reward, next_state, done)
                        # 使用经验回放更新DQN代理
                        updater.agent.replay(batch_size=64)

            print("Best val: ", best_val)
            print("Born Again...")
            random.seed(diff_seed[gen])
            np.random.seed(diff_seed[gen])
            torch.manual_seed(diff_seed[gen])

            # prepare for next generation of born again networks
            updater.register_last_model(last_model_weight, num_labels, model_select, device, dropout)
            updater.s_gen += 1
            best_val_list.append(best_val)
            best_val, best_val_loss = 0, 10000
            model, optimizer = model_utils.model_setup(num_labels, model_select, device, config, dropout)
            updater.model = model
            updater.optimizer = optimizer

            # get best temperature on dev set
            if args['calib_kd']:
                T = updater.get_calib_temp(valloader, y_val, device, loss_function, dataset)
                print("Temperature used to calibrate teacher predictions: {}".format(T))
            print("Best validation results: " + ",".join(map(str, best_val_list)))

        # 测试集评估
        for gen in range(s_gen, n_gen):
            # load the trained model
            weight = os.path.join(outdir, model_select + '_' + str(gen) + '_seed{}.pt'.format(seed))
            model.load_state_dict(torch.load(weight))

            model.eval()
            with torch.no_grad():
                preds, _ = model_utils.model_preds(testloader, model, device, loss_function)

                # micro-averaged F1
                f1_average = evaluation.compute_f1(preds, y_test, dataset)
                best_test_micro.append(f1_average)

                # macro-averaged F1
                preds_list = dahe.sep_test_set(preds, dataset)
                y_test_list = dahe.sep_test_set(y_test, dataset)
                temp_list = []
                for ind in range(len(y_test_list)):
                    f1_average = evaluation.compute_f1(preds_list[ind], y_test_list[ind], dataset)
                    temp_list.append(f1_average)
                best_test_macro.append(sum(temp_list) / len(temp_list))

        print("Best micro test results: " + ",".join(map(str, best_test_micro)))
        print("Best macro test results: " + ",".join(map(str, best_test_macro)))


if __name__ == "__main__":
    run_classifier()
