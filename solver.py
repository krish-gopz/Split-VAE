"""solver.py"""
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from utils import cuda, grid2gif
from model import BetaVAE_H, BetaVAE_B, Split_BetaVAE_G
from dataset import return_data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    #print(klds.size(), total_kld.size(), dimension_wise_kld.size(), mean_kld.size())
    #split_klds = dimension_wise_kld.split([24,8])
    #print(total_kld, split_klds[0].sum()+split_klds[1].sum())

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, parameters):
        self.use_cuda = parameters['use_cuda'] and torch.cuda.is_available()
        self.max_iter = parameters['max_iter']
        self.global_iter = 0
        if parameters['model'] == 'G':
            self.split_z_dim = parameters['split_z_dim']
            self.z_dim = sum(self.split_z_dim)
        else:
            self.z_dim = parameters['z_dim']
        self.beta = parameters['beta']
        self.objective = parameters['objective']
        self.model = parameters['model']
        self.lr = parameters['lr']
        self.beta1 = parameters['beta1']
        self.beta2 = parameters['beta2']
        if parameters['dataset'].lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif parameters['dataset'].lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif parameters['dataset'].lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if parameters['model'] == 'H':
          net = BetaVAE_H
        elif parameters['model'] == 'B':
          net = BetaVAE_B
        elif parameters['model'] == 'G':
          net = Split_BetaVAE_G
        else:
          raise NotImplementedError('only support model H or B')

        if parameters['model'] == 'G':
            self.net = cuda(net(self.split_z_dim, self.nc), self.use_cuda)
        else:
            self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        self.viz_name = parameters['project_name']
        self.viz_on = parameters['viz_on']
        self.tensorboard_dir = os.path.join(parameters['tensorboard_dir'], self.viz_name)
        self.ckpt_dir = os.path.join(parameters['ckpt_dir'], parameters['project_name'])
        #print(self.ckpt_dir)
        if not os.path.exists(self.ckpt_dir):
          os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = parameters['ckpt_name']
        if self.ckpt_name is not None:
          self.load_checkpoint(self.ckpt_name)
        if self.viz_on:
          self.viz = SummaryWriter(os.path.join(parameters['tensorboard_dir'], self.viz_name))
        self.save_output = parameters['save_output']
        self.output_dir = os.path.join(parameters['output_dir'], parameters['project_name'])
        if not os.path.exists(self.output_dir):
          os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = parameters['gather_step']
        self.display_step = parameters['display_step']
        self.save_step = parameters['save_step']

        self.dset_dir = parameters['dset_dir']
        self.dataset = parameters['dataset']
        self.batch_size = parameters['batch_size']
        self.data_loader = return_data(parameters)

        self.gather = DataGather()

    def train(self):
        self.net_mode(train=True)
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    # Change this later
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'G':
                    split_klds = dim_wise_kld.split(self.split_z_dim)
                    beta_vae_loss = recon_loss
                    for i, kldsplit in enumerate(split_klds):
                        beta_vae_loss = beta_vae_loss + self.beta[i]*kldsplit.sum()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                # Data gathering and display
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)
                if self.global_iter%self.display_step == 0:
                    #print(1)
                    #print(self.global_iter, recon_loss.data, total_kld.data[0], mean_kld.data[0])
                    #print(2)
                    #pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                    #    self.global_iter, recon_loss.data, total_kld.data[0], mean_kld.data[0]))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    #pbar.write(var_str)

                    if self.viz_on:
                      self.gather.insert(images=x.data)
                      self.gather.insert(images=F.sigmoid(x_recon).data)
                      self.viz_reconstruction()
                      self.viz_lines()
                      self.gather.flush()

                    if self.viz_on or self.save_output:
                      self.viz_traverse()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    #pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
        self.viz.close()

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True) # Check: range could be put for each dataset case
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True) # Check: range could be put for each dataset case
        #images = torch.stack([x, x_recon], dim=0).cpu()
        images = torch.stack([x, x_recon], dim=0).cpu()
        images = make_grid(images)
        #self.viz.images(images, env=self.viz_name+'_reconstruction',
        #                opts=dict(title=str(self.global_iter)), nrow=10)
        self.viz.add_image('Reconstruction', images, self.global_iter)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter']).tolist()
        #print(recon_losses.size(), mean_klds.size(), total_klds.size(), dim_wise_klds.size(), mus.size(), vars.size())
        recon_losses = recon_losses.tolist()
        # print(iters, recon_losses)
        for i in range(len(iters)):
            self.viz.add_scalar('reconstruction loss', recon_losses[i], iters[i])
            self.viz.add_scalar('Mean KL Divergence', mean_klds[i][0], iters[i])
            self.viz.add_scalar('Total KL Divergence', total_klds[i][0], iters[i])
            dim_wise_klds_dict = {}
            mus_dict = {}
            vars_dict = {}
            for z_dim in range(self.z_dim):
                dim_wise_klds_dict['z'+str(z_dim)] = dim_wise_klds[i][z_dim]
                mus_dict['z'+str(z_dim)] = mus[i][z_dim]
                vars_dict['z'+str(z_dim)] = vars[i][z_dim]
            self.viz.add_scalars('Dim-wise KL Divergence', dim_wise_klds_dict, iters[i])
            self.viz.add_scalars('Means', mus_dict, iters[i])
            self.viz.add_scalars('Variances', vars_dict, iters[i])
        # Check: ADD OTHER PLOTS HERE AS REQUIRED
        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)
        #print(interpolation.shape)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            #title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            title = '{}_latent_traversal'.format(key, self.global_iter)

            if self.viz_on:
                samples = make_grid(samples, nrow = len(interpolation))
                self.viz.add_image(title, samples, self.global_iter)

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        #win_states = {'recon':self.win_recon,
        #              'kld':self.win_kld,
        #              'mu':self.win_mu,
        #              'var':self.win_var,}
        win_states = None
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            #self.win_recon = checkpoint['win_states']['recon']
            #self.win_kld = checkpoint['win_states']['kld']
            #self.win_var = checkpoint['win_states']['var']
            #self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
            str1 = 'rm -rf '+self.tensorboard_dir
            subprocess.call(str1, shell=True)
            str1 = 'mkdir -p '+self.tensorboard_dir
            subprocess.call(str1, shell=True)

    def get_reconstructions(self, samples):
        self.net_mode(train=False)
        reconstructions = []
        if type(samples) != type([]):
            samples = [samples]
        for x in samples:
            x = Variable(cuda(x, self.use_cuda))
            x_recon, mu, logvar = self.net(x)
            reconstructions.append(F.sigmoid(x_recon).data)
        self.net_mode(train=True)
        return reconstructions
