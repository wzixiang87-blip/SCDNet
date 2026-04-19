from sympy import Ci
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.fft





warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):

        criterion = nn.SmoothL1Loss(beta=0.2)
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)[0]
                else:

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

                preds.append(pred.numpy())
                trues.append(true.numpy())

        total_loss = np.average(total_loss)


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mae, mse, _, _, _ = metric(preds, trues)

        self.model.train()

        print(f"Validation | Loss: {total_loss:.6f}, MSE: {mse:.6f}, MAE: {mae:.6f}")
        return mse

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.lradj == 'type3':
            scheduler = optim.lr_scheduler.MultiStepLR(model_optim, milestones=[5, 10, 15, 20], gamma=0.1)

        is_sam_optimizer = hasattr(model_optim, 'first_step') and hasattr(model_optim, 'second_step')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []


            log_losses = {'time': [], 'freq': [], 'aux': []}

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                def get_loss_components():
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                    else:
                        model_output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)

                    aux_loss = torch.tensor(0.0, device=self.device)
                    if isinstance(model_output, tuple):
                        outputs = model_output[0]

                        if len(model_output) >= 2 and model_output[1] is not None:
                            aux_loss = model_output[1]

                            if aux_loss.ndim > 0: aux_loss = aux_loss.mean()
                    else:
                        outputs = model_output

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_slice = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                    time_loss = criterion(outputs, batch_y_slice)


                    pred_fft = torch.fft.rfft(outputs, dim=1)
                    true_fft = torch.fft.rfft(batch_y_slice, dim=1)

                    freq_loss = criterion(torch.view_as_real(pred_fft), torch.view_as_real(true_fft))


                    w_time, w_freq, w_aux = 1.0, 0.4, 0.01
                    total_loss = w_time * time_loss + w_freq * freq_loss + w_aux * aux_loss

                    return total_loss, time_loss, freq_loss, aux_loss


                if is_sam_optimizer:

                    loss, t_l, f_l, a_l = get_loss_components()
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(model_optim)
                    else:
                        loss.backward()


                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                    model_optim.first_step(zero_grad=True)

                    # SAM Step 2
                    loss_2, _, _, _ = get_loss_components()
                    if self.args.use_amp:
                        scaler.scale(loss_2).backward()
                        model_optim.second_step(zero_grad=True)

                        scaler.update()
                    else:
                        loss_2.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                        model_optim.second_step(zero_grad=True)

                else:

                    loss, t_l, f_l, a_l = get_loss_components()

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(model_optim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                        model_optim.step()


                train_loss.append(loss.item())
                log_losses['time'].append(t_l.item())
                log_losses['freq'].append(f_l.item())
                log_losses['aux'].append(a_l.item())

                if (i + 1) % 100 == 0:
                    avg_t = np.average(log_losses['time'][-100:])
                    avg_f = np.average(log_losses['freq'][-100:])
                    avg_a = np.average(log_losses['aux'][-100:])

                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | Total: {loss.item():.7f} "
                          f"| T-Loss: {avg_t:.5f} | F-Loss: {avg_f:.5f} | Aux-Loss: {avg_a:.5f}")

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)


            vali_mse = self.vali(vali_data, vali_loader, criterion)


            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali MSE: {vali_mse:.7f}")


            early_stopping(vali_mse, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj == 'type1':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            elif self.args.lradj == 'type3':
                scheduler.step()
                current_lr = model_optim.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}: learning rate = {current_lr:.6f}")


        torch.save(early_stopping.checkpoint_in_gpu, path + '/' + 'checkpoint.pth')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_cycle)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return