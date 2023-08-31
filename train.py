import torch
import numpy as np
import argparse
import time
from Util_general import *
from engine import *
from paths import *
import os
import shutil
import random

def run_model():
    # load data
    data_path = ''
    save_path = ''
    if DATASET == 'PEMS-BAY':
        data_path = DPATH1
        save_path = SPATH1
    elif DATASET == 'PEMS08':
        data_path = DPATH3
        save_path = SPATH3

    dataloader = load_dataset_with_cluster(data_path, BATCH_SIZE)
    pre_supports_list = generate_support_list(data_path, GRAPH_TYPE, PROCESSING_TYPE)
    mean, std = pickle.load(open(data_path + '\\' + 'mean_and_std.pickle', 'rb'))
    transmit = pickle.load(open(data_path + '\\' + 'transmit.pickle', 'rb'))
    spa_embedding = pickle.load(open(data_path + '\\' + 'spatial_embedding.pickle', 'rb'))

    supports_list = []
    for supports in pre_supports_list:
        supports_list.append([torch.tensor(support).to(DEVICE) for support in supports])
    transmit = torch.tensor(np.float32(transmit)).to(DEVICE)
    spa_embedding = torch.tensor(spa_embedding)
    spa_embedding = spa_embedding.type(torch.FloatTensor)
    spa_embedding = spa_embedding.to(DEVICE)
    mean1 = torch.tensor(mean).to(DEVICE)
    std1 = torch.tensor(std).to(DEVICE)

    num_nodes = dataloader['x_train'].shape[2]
    num_regions = dataloader['x_c_train'].shape[2]
    num_graphs = len(supports_list)
    num_time_steps = 12


    if MODEL == 'HST':
        engine = trainer0(DEVICE, num_nodes, num_regions, num_graphs, IN_DIM, IN_DIM_CLUSTER, OUT_DIM, CHANNELS_0,
                          CHANNELS_1, CHANNELS_2, supports_list, transmit, MF_EMBEDDING_SIZE, num_time_steps, KT, K_TA,
                          ORDER_T, K_SA, L_RATE, WDECAY, DECAY, mean, std)

    params_name = str(EXP_ID) + '_' + str(IN_DIM) + '_' + str(IN_DIM_CLUSTER) + '_' + str(OUT_DIM) + '_' + str(CHANNELS_0) + '_' + \
                  str(CHANNELS_1) + '_' + str(CHANNELS_2) + '_' + str(MF_EMBEDDING_SIZE) + '_' + str(L_RATE) + '_' + \
                  str(IN_DIM) + '_' + str(KT) + '_' + str(K_TA) + '_' +  str(ORDER_T) + '_' + str(K_SA) + '_' + \
                  str(GRAPH_TYPE) + '_' + str(PROCESSING_TYPE) + '_' + str(DROPOUT)
    params_path = save_path + '\\' + MODEL + '\\' + params_name
    if os.path.exists(params_path) and not FORCE:
        raise SystemExit("Params folder exists!")
    else:
        if os.path.exists(params_path):
            shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('Create params directory %s' % (params_path))

    total_valid_loss = []
    for epoch in range(1, EPOCHS+1):
        # training
        te1 = time.time()
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        dataloader['train_loader'].shuffle()
        for i, (x, y, x_c, tem_embedding) in enumerate(dataloader['train_loader'].get_iterator()):
            train_x = torch.Tensor(x).to(DEVICE)
            train_y = torch.Tensor(y).to(DEVICE)
            train_x_c = torch.Tensor(x_c).to(DEVICE)
            train_tem_em = torch.Tensor(tem_embedding).to(DEVICE)
            metrics = engine.train(train_x, train_x_c, train_y, train_tem_em, spa_embedding)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_mape.append(metrics[2])
            train_rmse.append(metrics[3])
        # engine.scheduler.step()

        # validation
        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        for i, (x, y, x_c, tem_embedding) in enumerate(dataloader['valid_loader'].get_iterator()):
            valid_x = torch.Tensor(x).to(DEVICE)
            valid_y = torch.Tensor(y).to(DEVICE)
            valid_x_c = torch.Tensor(x_c).to(DEVICE)
            valid_tem_em = torch.Tensor(tem_embedding).to(DEVICE)
            metrics = engine.eval(valid_x, valid_x_c, valid_y, valid_tem_em, spa_embedding)
            valid_loss.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            valid_rmse.append(metrics[3])

        avg_train_loss = np.mean(train_loss)
        avg_train_mae = np.mean(train_mae)
        avg_train_mape = np.mean(train_mape)
        avg_train_rmse = np.sqrt(np.mean(np.array(train_rmse)**2))
        avg_valid_loss = np.mean(valid_loss)
        avg_valid_mae = np.mean(valid_mae)
        avg_valid_mape = np.mean(valid_mape)
        avg_valid_rmse = np.sqrt(np.mean(np.array(valid_rmse)**2))
        total_valid_loss.append(avg_valid_loss)

        te2 = time.time()
        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training time: {:.4f}'
        print(log.format(epoch, avg_train_loss, avg_train_mae, avg_train_mape, avg_train_rmse, avg_valid_loss,
                         avg_valid_mae, avg_valid_mape, avg_valid_rmse, round(te2-te1, 3), flush=True))
        torch.save(engine.model.state_dict(), params_path + '\\' + 'epoch_' + str(epoch) + '_' + str(round(avg_valid_loss, 2)) + '.pth')

    # testing
    optimal_id = np.argmin(total_valid_loss)
    engine.model.load_state_dict(torch.load(
        params_path + '\\' + 'epoch_' + str(optimal_id+1) + '_' + str(round(total_valid_loss[optimal_id], 2)) + '.pth'))
    engine.model.eval()

    test_outputs = []
    test_y = torch.Tensor(dataloader['y_test']).to(DEVICE)
    for i, (x, y, x_c, tem_embedding) in enumerate(dataloader['test_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(DEVICE)
        test_x_c = torch.Tensor(x_c).to(DEVICE)
        test_tem_em = torch.Tensor(tem_embedding).to(DEVICE)
        with torch.no_grad():
            preds = engine.model(test_x, test_x_c, test_tem_em, spa_embedding, mean1, std1)
        test_outputs.append(preds)
    test_y_pred = torch.cat(test_outputs, dim=0)
    test_y_pred = test_y_pred[:test_y.size(0), ...]
    test_y_pred = test_y_pred.transpose(1, 3)

    print("Training finished")
    print("The valid loss on best model under current hyper params is", str(round(total_valid_loss[optimal_id], 4)))

    smae = []
    smape = []
    srmse = []
    for i in range(num_time_steps):
        pred = test_y_pred[:, :, :, i]
        real = test_y[:, :, :, i]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        smae.append(metrics[0])
        smape.append(metrics[1])
        srmse.append(metrics[2])
    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(smae), np.mean(smape), np.sqrt(np.mean(np.array(srmse)**2))))

    metrics2 = metric(test_y_pred, test_y)
    print(log.format(metrics2[0], metrics2[1], metrics2[2]))

    torch.save(engine.model.state_dict(),
               params_path + '\\' + 'best_epoch_' + str(optimal_id+1) + '_' + str(round(total_valid_loss[optimal_id], 2)) + '.pth')
    results = {}
    results['test_result_separate'] = [smae, smape, srmse]
    results['test_result_total'] = metrics2
    results['test_prediction'] = test_y_pred.cpu().detach().numpy()
    results['test_ground_truth'] = test_y.cpu().detach().numpy()

    pickle.dump(results, open(params_path + '\\' +  "test_results.pickle", 'wb'))

    return total_valid_loss[optimal_id]


if __name__ == "__main__":
    # hyperparameters
    DEVICE = torch.device('cuda:0')
    IN_DIM = 1
    IN_DIM_CLUSTER = 3
    OUT_DIM = 12
    CHANNELS_0 = 64
    CHANNELS_1 = 128
    CHANNELS_2 = 256
    MF_EMBEDDING_SIZE = 4
    KT = 3
    K_TA = 8
    ORDER_T = 3
    K_SA = 8
    L_RATE = 0.001
    WDECAY = 0.000
    DECAY = 1.
    DROPOUT = 0.
    BATCH_SIZE = 64
    EPOCHS = 50
    EXP_ID = 0
    DATASET = 'PEMS08'
    GRAPH_TYPE = 'distance_and_speed_similarity'
    PROCESSING_TYPE = 'double_transition'
    MODEL = 'HST'
    FORCE = True

    # HST is the MMSTNet
    run_model()














