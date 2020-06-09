import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

DEVICE = 'cuda'

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(100, 100)

        self.init_size = 32 // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def init_weights(self):
        self.apply(weights_init_normal)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 100), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

    def init_weights(self):
        self.apply(weights_init_normal)


class ACGAN():
    def __init__(self):
        self.gen = Generator()
        self.gen.init_weights()
        self.gen.to(DEVICE)

        self.discr = Discriminator()
        self.discr.init_weights()
        self.discr.to(DEVICE)

        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.discr_opt = torch.optim.Adam(self.discr.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.adv_loss = torch.nn.BCELoss()
        self.aux_loss = torch.nn.CrossEntropyLoss()  

    def train(self, loader):
        for epoch in range(2):
            for images, labels in loader:
                batch_size = images.shape[0]

                # Adversarial ground truths
                # valid = torch.FloatTensor(batch_size, 1, requires_grad=False).fill_(1.0)
                valid = torch.ones(batch_size, 1, requires_grad=False, dtype=torch.float).to(DEVICE)
                # fake = torch.FloatTensor(batch_size, 1, requires_grad=False).fill_(0.0)
                fake = torch.zeros(batch_size, 1, requires_grad=False, dtype=torch.float).to(DEVICE)

                # Configure input
                real_imgs = torch.FloatTensor(images).to(DEVICE)
                labels = torch.LongTensor(labels).to(DEVICE)

                # -----------------
                #  Train Generator
                # -----------------

                self.gen_opt.zero_grad()

                # Sample noise and labels as generator input
                z = torch.FloatTensor(np.random.normal(0, 1, (batch_size, 100))).to(DEVICE)
                gen_labels = torch.LongTensor(np.random.randint(0, 100, batch_size)).to(DEVICE)

                # Generate a batch of images
                gen_imgs = self.gen(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity, pred_label = self.discr(gen_imgs)
                g_loss = 0.5 * (self.adv_loss(validity, valid) + self.aux_loss(pred_label, gen_labels))

                g_loss.backward()
                self.gen_opt.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.discr_opt.zero_grad()

                # Loss for real images
                real_pred, real_aux = self.discr(real_imgs)
                d_real_loss = (self.adv_loss(real_pred, valid) + self.aux_loss(real_aux, labels)) / 2

                # Loss for fake images
                fake_pred, fake_aux = self.discr(gen_imgs.detach())
                d_fake_loss = (self.adv_loss(fake_pred, fake) + self.aux_loss(fake_aux, gen_labels)) / 2

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                # Calculate discriminator accuracy
                pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
                gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
                d_acc = np.mean(np.argmax(pred, axis=1) == gt)

                d_loss.backward()
                self.discr_opt.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
                    % (epoch, 100, 2, len(loader), d_loss.item(), 100 * d_acc, g_loss.item())
                )