import os
import matplotlib
import matplotlib.pyplot as plt

from datasets.dataset_fetcher import DatasetFetcher

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.models as models
from utils import common, train_utils
from criteria import id_loss
from configs import data_conf
from datasets.images_dataset import ImagesDataset
from datasets.latents_images_dataset import LatentsImagesDataset
from criteria.lpips.lpips import LPIPS
from models.hifisketch import Hifisketch
from training.ranger import Ranger
import numpy as np
from torch.nn import init

class Dis_conv(nn.Module):
    def __init__(self):
        super(Dis_conv, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=4, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.model(x)

class Psim(nn.Module):
    def __init__(self, patch_nums=256, patch_size=None, norm=True):
        super(Psim, self).__init__()
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.use_norm = norm

    def forward(self, feat, patch_ids=None):
        B, C, W, H = feat.size()
        feat = feat - feat.mean(dim=[-2, -1], keepdim=True)
        feat = F.normalize(feat, dim=1) if self.use_norm else feat / np.sqrt(C)
        query, key, patch_ids = self.sele_p(feat, p_num=patch_ids)
        patch_sim = query.bmm(key) if self.use_norm else torch.tanh(query.bmm(key)/10)
        if patch_ids is not None:
            patch_sim = patch_sim.view(B, len(patch_ids), -1)

        return patch_sim, patch_ids

    def sele_p(self, feat, p_num=None):
        B, C, W, H = feat.size()
        pw, ph = self.patch_size, self.patch_size
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2) # B*N*C
        if self.patch_nums > 0:
            if p_num is None:
                p_num = torch.randperm(feat_reshape.size(1), device=feat.device)
                p_num = p_num[:int(min(self.patch_nums, p_num.size(0)))]
            fea_que = feat_reshape[:, p_num, :]       # B*Num*C
            fea_key = []
            Num = fea_que.size(1)
            if pw < W and ph < H:
                pos_x, pos_y = p_num // W, p_num % W
                left, top = pos_x - int(pw / 2), pos_y - int(ph / 2)
                left, top = torch.where(left > 0, left, torch.zeros_like(left)), torch.where(top > 0, top, torch.zeros_like(top))
                start_x = torch.where(left > (W - pw), (W - pw) * torch.ones_like(left), left)
                start_y = torch.where(top > (H - ph), (H - ph) * torch.ones_like(top), top)
                for i in range(Num):
                    fea_key.append(feat[:, :, start_x[i]:start_x[i]+pw, start_y[i]:start_y[i]+ph]) # B*C*patch_w*patch_h
                fea_key = torch.stack(fea_key, dim=0).permute(1, 0, 2, 3, 4) # B*Num*C*patch_w*patch_h
                fea_key = fea_key.reshape(B * Num, C, pw * ph)  # Num * C * N
                fea_que = fea_que.reshape(B * Num, 1, C)  # Num * 1 * C
            else: # if patch larger than features size, use B * C * N (H * W)
                fea_key = feat.reshape(B, C, W*H)
        else:
            fea_que = feat.reshape(B, C, H*W).permute(0, 2, 1) # B * N (H * W) * C
            fea_key = feat.reshape(B, C, H*W)  # B * C * N (H * W)

        return fea_que, fea_key, p_num

def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net

class cross_domain_loss(nn.Module):
    def __init__(self, loss_mode='cos', patch_nums=256, patch_size=32, norm=True, use_conv=True,
                 init_type='normal', init_gain=0.02, gpu_ids=[], T=0.1):
        super(cross_domain_loss, self).__init__()
        self.patch_sim = Psim(patch_nums=patch_nums, patch_size=patch_size, norm=norm)
        self.patch_size = patch_size
        self.patch_nums = patch_nums
        self.norm = norm
        self.use_conv = use_conv
        self.conv_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.loss_mode = loss_mode
        self.T = T
        self.criterion = nn.L1Loss() if norm else nn.SmoothL1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cr_conv(self, feat, layer):
        input_nc = feat.size(1)
        output_nc = max(32, input_nc // 4)
        conv = nn.Sequential(*[nn.Conv2d(input_nc, output_nc, kernel_size=1),
                               nn.ReLU(),
                               nn.Conv2d(output_nc, output_nc, kernel_size=1)])
        conv.to(feat.device)
        setattr(self, 'conv_%d' % layer, conv)
        init_net(conv, self.init_type, self.init_gain, self.gpu_ids)

    def calc_sim(self, f_src, f_tgt, f_other=None, layer=0, p_num=None):
        if self.use_conv:
            if not self.conv_init:
                self.cr_conv(f_src, layer)
            conv = getattr(self, 'conv_%d' % layer)
            f_src, f_tgt = conv(f_src), conv(f_tgt)
            f_other = conv(f_other) if f_other is not None else None
        sim_src, p_num = self.patch_sim(f_src, p_num)
        sim_tgt, p_num = self.patch_sim(f_tgt, p_num)
        if f_other is not None:
            sim_other, _ = self.patch_sim(f_other, p_num)
        else:
            sim_other = None

        return sim_src, sim_tgt, sim_other

    def com_sim(self, sim_src, sim_tgt, sim_other):
        B, Num, N = sim_src.size()
        if self.loss_mode == 'info' or sim_other is not None:
            sim_src = F.normalize(sim_src, dim=-1)
            sim_tgt = F.normalize(sim_tgt, dim=-1)
            sim_other = F.normalize(sim_other, dim=-1)
            sam_neg1 = (sim_src.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_neg2 = (sim_tgt.bmm(sim_other.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = (sim_src.bmm(sim_tgt.permute(0, 2, 1))).view(-1, Num) / self.T
            sam_self = torch.cat([sam_self, sam_neg1, sam_neg2], dim=-1)
            loss = self.cross_entropy_loss(sam_self, torch.arange(0, sam_self.size(0), dtype=torch.long, device=sim_src.device) % (Num))
        else:
            tgt_sorted, _ = sim_tgt.sort(dim=-1, descending=True)
            num = int(N / 4)
            source = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_src, sim_src)
            target = torch.where(sim_tgt < tgt_sorted[:, :, num:num + 1], 0 * sim_tgt, sim_tgt)
            if self.loss_mode == 'l1':
                loss = self.criterion((N / num) * source, (N / num) * target)
            elif self.loss_mode == 'cos':
                sim_pos = F.cosine_similarity(source, target, dim=-1)
                loss = self.criterion(torch.ones_like(sim_pos), sim_pos)
            else:
                raise NotImplementedError('padding [%s] is not implemented' % self.loss_mode)

        return loss

    def loss(self, source_f, target_f, other_f=None, layer=0):
        source_sim, target_sim, o_sim = self.calc_sim(source_f, target_f, other_f, layer)
        loss = self.com_sim(source_sim, target_sim, o_sim)
        return loss



class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network
        self.net = Hifisketch(self.opts).to(self.device)
        self.Diconv=Dis_conv().to(self.device)
        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()


        # Initialize loss

        self.mse_loss = nn.MSELoss().to(self.device).eval()
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss(opts).to(self.device).eval()
        self.Dis_loss=nn.BCELoss()
        self.cd_loss=cross_domain_loss()



        # Initialize optimizer
        self.optimizer = self.configure_optimizers()
        self.Dis_optimizer = torch.optim.Adam(self.Diconv.parameters(), lr=self.opts.learning_rate)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def parse_batch(self, batch):
        x, y, y_hat, latents = None, None, None, None
        if isinstance(self.train_dataset, ImagesDataset):
            x, y = batch
            x, y = x.to(self.device).float(), y.to(self.device).float()
        elif isinstance(self.train_dataset, LatentsImagesDataset):
            y_hat, y, latents = batch  # source (inversion), target, and latent code
            y_hat, y, latents = y_hat.to(self.device).float(), y.to(self.device).float(), latents.to(self.device)
            x = y
        else:
            raise ValueError("Unsupported dataset type")
        return x, y, y_hat, latents

    def perform_forward_on_batch(self, batch, train=False):
        latent, weights_deltas, w_inversion, initial_inversion = None, None, None, None
        cur_loss_dict, id_logs = None, None
        x, y, y_hat, codes = self.parse_batch(batch)
        y_hats = {idx: [] for idx in range(x.shape[0])}
        for iter in range(self.opts.n_iters_per_batch):
            if iter > 0 and train:
                weights_deltas = [w.clone().detach().requires_grad_(True) if w is not None else w
                                  for w in weights_deltas]
                y_hat = y_hat.clone().detach().requires_grad_(True)
            y_hat, latent, weights_deltas, codes, w_inversion = self.net.forward(x,
                                                                                 y_hat=y_hat,
                                                                                 codes=codes,
                                                                                 weights_deltas=weights_deltas,
                                                                                 return_latents=True,
                                                                                 randomize_noise=False,
                                                                                 return_weight_deltas_and_codes=True,
                                                                                 resize=True)
            if iter == 0:
                initial_inversion = w_inversion

            loss, cur_loss_dict, id_logs = self.calc_loss(x=y,
                                                          y=y,
                                                          y_hat=y_hat,
                                                          latent=latent,
                                                          weights_deltas=weights_deltas)
            if train:
                loss.backward()

            # store intermediate outputs
            for idx in range(x.shape[0]):
                y_hats[idx].append([y_hat[idx].detach().cpu(), id_logs[idx]['diff_target']])
        return x, y, y_hats,y_hat, cur_loss_dict, id_logs, initial_inversion

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                self.Dis_optimizer.zero_grad()
                x, y, y_hat ,y_hat_gpu, loss_dict, id_logs, w_inversion = self.perform_forward_on_batch(batch, train=True)
                Dis_pred = self.Diconv(y_hat_gpu.clone())
                self.optimizer.step()
                self.Dis_optimizer.step()
                cd_loss=self.cd_loss.loss(x, y_hat_gpu)
                G_loss=loss_dict['loss']+cd_loss

                self.Dis_optimizer.zero_grad()
                Dis_pred = self.Diconv(y_hat_gpu)
                pred_label = torch.zeros_like(Dis_pred)
                Dis_real = self.Diconv(y)
                real_label = torch.ones_like(Dis_real)
                DisLoss = self.Dis_loss(Dis_pred,pred_label)*(1/2)+self.Dis_loss(Dis_real,real_label)*(1/2)

                loss_dict['loss'] = G_loss+ DisLoss.item()

                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')


                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step > 0 and (self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('finished training!!!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            if self.opts.max_val_batches is not None and batch_idx > self.opts.max_val_batches:
                break
            with torch.no_grad():
                x, y, y_hat,y_hat_gpu, cur_loss_dict, id_logs, w_inversion = self.perform_forward_on_batch(batch)
            agg_loss_dict.append(cur_loss_dict)



            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.hifinet.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_conf.DATASETS.keys():
            raise Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_conf.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset, test_dataset = DatasetFetcher().get_dataset(self.opts, dataset_args, transforms_dict)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, weights_deltas):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if "cars" in self.opts.dataset_type:
            y_hat_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
            y_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y)
            x_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(x)
        else:
            y_hat_resized = self.net.face_pool(y_hat)
            y_resized = self.net.face_pool(y)
            x_resized = self.net.face_pool(x)

        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat_resized, y_resized, x_resized)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y)
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat_resized, y_resized)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda


        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)


    def __get_save_dict(self):
        save_dict = {
            'state_dict': {k: v for k, v in self.net.state_dict().items() if 'w_encoder' not in k},
            'opts': vars(self.opts),
            'latent_avg': self.net.latent_avg,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        return save_dict

    def __load_train_checkpoint(self, checkpoint):
        print('Loading previous training data...')
        self.global_step = checkpoint['global_step'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        # self.net.load_state_dict(checkpoint['state_dict'])
        print(f'Resuming training from step: {self.global_step}')
        print(f'Current best validation loss: {self.best_val_loss}')
