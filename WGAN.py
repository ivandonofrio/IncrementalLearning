import numpy as np
import math
import time

from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch
from torch.autograd import Variable

DEVICE = 'cuda'

'''
Inspired by:    https://github.com/xialeiliu/GFR-IL/blob/master/models/resnet.py
                https://github.com/haseebs/Pseudo-rehearsal-Incremental-Learning/blob/master/model/WGAN.py
'''

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def compute_gradient_penalty(D, real_samples, fake_samples, syn_label):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates, syn_label)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=fake,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

class Generator(nn.Module):
    def __init__(self, norm=False, latent_dim=200, class_dim=100, feat_dim=64, hidden_dim=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim + class_dim
        self.feat_dim = feat_dim
        self.norm = norm
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.apply(weights_init)

    def forward(self, z, label):
        z = torch.cat((z, label), 1)
        img = self.model(z)
        if self.norm:
            img = F.normalize(img)
        return img

class Discriminator(nn.Module):
    def __init__(self, feat_dim=64, class_dim=100, hidden_dim=64, condition='projection'):
        super(Discriminator, self).__init__()

        self.feat_dim = feat_dim
        self.model = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.valid = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)
        self.classifer = nn.Linear(hidden_dim, class_dim)
        self.projection = nn.Linear(class_dim, hidden_dim, bias = False)
        self.condition = condition

    def forward(self, img, label):
        hidden = self.model(img)
        var = self.projection(label)
        validity = (var * hidden).sum(dim=1).reshape(-1, 1) + self.valid(hidden)
        classifier = self.classifer(hidden)
        return validity, classifier


class WGAN():
    def __init__(self, parameters, num_classes):
        self.gen = Generator(class_dim=num_classes)
        self.gen = self.gen.to(DEVICE)

        self.discr = Discriminator(class_dim=num_classes)
        self.discr = self.discr.to(DEVICE)

        self.num_classes = num_classes
        self.parameters = parameters

    def generate_examples(self, num_examples, active_classes):
        '''
        Returns a dict[class] of generated samples.
        Just passing in random noise to the generator and storing the results in dict
        Generates a batch of 100 examples at a time
        num_examples: Total number of examples to generate
        active_classes: List of all classes trained on till now
        '''
        with torch.no_grad():
            self.gen.eval()
            #for param in G.parameters():
            #        param.requires_grad = False
            features = []
            labels = []

            for klass in active_classes:
                # One-hot encoding
                targets = torch.zeros(num_examples, self.num_classes)
                targets[:, klass] = 1
                # Random noise
                z = torch.Tensor(np.random.normal(0, 1, (num_examples, 200))).to(DEVICE)
                targets = targets.to(DEVICE)

                out = self.gen(z, targets)
                if len(features) == 0:
                    features = out.cpu()
                else:
                    features = torch.cat((features, out.cpu()))
                if len(labels) == 0:
                    labels = torch.ones(num_examples) * klass
                else:
                    labels = torch.cat((labels, torch.ones(num_examples) * klass))
            
        return features, labels

    def train(self, loader, learned_classes, model):

        gen_params = self.parameters['GEN_PARAMETERS']
        discr_params = self.parameters['DISCR_PARAMETERS']
        opt_gen = gen_params['OPTIMIZER'](self.gen.parameters(), **gen_params['OPTIMIZER_PARAMETERS'])
        opt_discr = discr_params['OPTIMIZER'](self.discr.parameters(), **discr_params['OPTIMIZER_PARAMETERS'])
        sched_gen = gen_params['SCHEDULER'](opt_gen, **gen_params['SCHEDULER_PARAMETERS'])
        sched_discr = discr_params['SCHEDULER'](opt_discr, **discr_params['SCHEDULER_PARAMETERS'])

        criterion_softmax = nn.CrossEntropyLoss().to(DEVICE)
        loss_mse = nn.MSELoss(reduction='sum')

        for epoch in range(self.parameters['NUM_EPOCHS']):
            mean_g = 0
            d_losses_e = []
            g_losses_e = []

            start = time.time()

            for i, (images, labels) in enumerate(loader):
                self.gen.train()
                self.discr.train()

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                y_onehot = torch.FloatTensor(len(labels), self.num_classes)

                ### Train discriminator ###
                opt_discr.zero_grad()
                real_feat = model(images, get_only_features=True)
                z = torch.Tensor(np.random.normal(0, 1, (len(labels), 200))).to(DEVICE)

                y_onehot.zero_()
                y_onehot.cuda().scatter_(1, labels[:, None], 1)
                syn_label = y_onehot.to(DEVICE)
                fake_feat = self.gen(z, syn_label)
                fake_validity, _               = self.discr(fake_feat, syn_label)
                real_validity, disc_real_acgan = self.discr(real_feat, syn_label)

                # Adversarial loss
                d_loss_rf = torch.mean(fake_validity) - torch.mean(real_validity)
                gradient_penalty = compute_gradient_penalty(self.discr, real_feat, fake_feat, syn_label).mean()
                d_loss_lbls = criterion_softmax(disc_real_acgan, labels)
                d_loss = d_loss_rf + self.parameters['LAMBDA_GRADIENT'] * gradient_penalty

                d_loss.backward()
                opt_discr.step()
                d_losses_e.append(d_loss.cpu().data.numpy())

                ### Train generator ###
                if i % self.parameters['DISCR_ITER'] == 0:
                    self.discr.eval()

                    opt_gen.zero_grad()
                    fake_feat = self.gen(z, syn_label)

                    fake_validity, disc_fake_acgan = self.discr(fake_feat, syn_label)
                    if model.iterations == 0:
                        loss_aug = 0 * torch.sum(fake_validity)
                    else:

                        embed_label_sythesis = torch.from_numpy(np.random.choice(list(learned_classes), len(labels), replace=True)).cuda()
                        y_onehot.zero_()
                        y_onehot.cuda().scatter_(1, embed_label_sythesis[:, None], 1)
                        syn_label_pre = y_onehot.cuda()

                        pre_feat = self.gen(z, syn_label_pre)
                        pre_feat_old = self.old_gen(z, syn_label_pre)
                        loss_aug = loss_mse(pre_feat, pre_feat_old)

                    g_loss_rf = - torch.mean(fake_validity)
                    g_loss_lbls = criterion_softmax(disc_fake_acgan, labels.cuda())
                    g_loss = g_loss_rf + self.parameters['LAMBDA_LWF'] * model.iterations * loss_aug

                    g_loss.backward()
                    opt_gen.step()
                    g_losses_e.append(g_loss.cpu().data.numpy())
            
            sched_gen.step()
            sched_discr.step()

            # Stats
            time_taken = time.time() - start
            if len(g_losses_e) > 0:
                mean_g = (sum(g_losses_e)/len(g_losses_e))
            mean_d = (sum(d_losses_e)/len(d_losses_e))
            print("[GAN] Epoch: {}, g_loss: {}, d_loss: {}, time taken: {}".format(
                  epoch + 1, mean_g, mean_d, time_taken
              ))

        print('Example of generated features:', fake_feat[0])