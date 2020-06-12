import numpy as np
import math
import time

from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

DEVICE = 'cuda'

'''
Inspired by: https://github.com/haseebs/Pseudo-rehearsal-Incremental-Learning/blob/master/model/cDCGAN.py
'''

def normal_init(m, mean, std, has_bias=True):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if has_bias:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    '''
    d = base multiplier
    c = number of channels in the image
    l = number of unique classes in the dataset
    '''
    def __init__(self, d=64, c=3, l=10):
        super(Generator, self).__init__()
        #ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.ct1_noise = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.ct1_noise_bn = nn.BatchNorm2d(d*2)
        self.ct1_label = nn.ConvTranspose2d(l, d*2, 4, 1, 0)
        self.ct1_label_bn = nn.BatchNorm2d(d*2)
        self.ct2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.ct2_bn = nn.BatchNorm2d(d*2)
        self.ct3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.ct3_bn = nn.BatchNorm2d(d)
        self.ct4 = nn.ConvTranspose2d(d, c, 4, 2, 1)

    def forward(self, noise, label):
        x = F.relu(self.ct1_noise_bn(self.ct1_noise(noise)))
        y = F.relu(self.ct1_label_bn(self.ct1_label(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.ct2_bn(self.ct2(x)))
        x = F.relu(self.ct3_bn(self.ct3(x)))
        x = torch.tanh(self.ct4(x))
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class Discriminator(nn.Module):
    '''
    d = base multiplier
    c = number of channels in the image
    l = number of unique classes in the dataset
    '''
    def __init__(self, d=64, c=3, l=10, use_mbd=False, mbd_num=128, mbd_dim=3):
        super(Discriminator, self).__init__()
        self.use_mbd = use_mbd
        self.mbd_num = mbd_num
        self.mbd_dim = mbd_dim
        self.d = d

        self.conv1_img = nn.Conv2d(c, d//2, 4, 2, 1)
        self.conv1_label = nn.Conv2d(l, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        # Use linear layer to produce a matrix of shape (B, mbd_num*mbd_dim)
        # The input to this layer is of shape [B, d*4, 4, 4], we reshape it
        if self.use_mbd:
            self.mbd = nn.Linear(d*4*4*4, mbd_num * mbd_dim)
            self.conv4 = nn.Conv2d(d * 4 + 8, 1, 4, 1, 0)
        else:
            self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    def forward(self, img, label):
        x = F.leaky_relu(self.conv1_img(img), 0.2)
        y = F.leaky_relu(self.conv1_label(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        if self.use_mbd:
            # Reshape for linear layer
            x = x.view(-1, self.d * 4 * 4 * 4)
            # Use the linear layer
            mbd = self.mbd(x)
            # Calculate minibatch features and concat them
            x = self.minibatch_discrimination(mbd, x)
            # Make it compatible for convolution layer
            # 520 = 128 * 4 + 8
            x = x.view(-1, self.d * 4 + 8, 4, 4)
        x = self.conv4(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def minibatch_discrimination(self, x, input_to_layer):
        activation = x.view(-1, self.mbd_num, self.mbd_dim)
        diffs = activation.unsqueeze(3) - activation.permute(1,2,0).unsqueeze(0)
        abs_diff = torch.abs(diffs).sum(2)
        mb_feats = torch.exp(-abs_diff).sum(2)
        return torch.cat([input_to_layer, mb_feats], 1)


class cDCGAN():
    def __init__(self, parameters, num_classes):
        self.num_classes = num_classes

        self.gen = Generator(l=num_classes)
        self.gen.init_weights(mean=0.0, std=0.02)
        self.gen = self.gen.to(DEVICE)

        # Enable mini-batch discrimination
        self.discr = Discriminator(l=num_classes, use_mbd=True)
        self.discr.init_weights(mean=0.0, std=0.02)
        self.discr = self.discr.to(DEVICE)

        self.criterion = parameters['CRITERION']()
        self.parameters = parameters

    def generate_examples(self, num_examples, active_classes, save=False):
        '''
        Returns a dict[class] of generated samples.
        Just passing in random noise to the generator and storing the results in dict
        Generates a batch of 10 examples at a time
        num_examples: Total number of examples to generate
        active_classes: List of all classes trained on till now
        save: If True, also save samples of generated images to disk
        '''
        with torch.no_grad():
            self.gen.eval()
            #for param in G.parameters():
            #        param.requires_grad = False
            examples = {}
            num_iter = 0
            for klass in active_classes:
                while ((not klass in examples.keys()) or (len(examples[klass]) < num_examples)):
                    # print(f'Generating for class {klass}, iteration {num_iter}')
                    num_iter += 1

                    targets = torch.zeros(100, self.num_classes, 1, 1)
                    targets[:, klass] = 1
                    noise = torch.randn(100, 100, 1, 1).to(DEVICE)
                    targets = targets.to(DEVICE)

                    images = self.gen(noise, targets)

                    if not klass in examples.keys():
                        examples[klass] = images.cpu()
                    else:
                        examples[klass] = torch.cat((examples[klass],images.cpu()), dim=0)

                # Store images locally
                if save:
                    for i,img in enumerate(examples[klass]):
                        save_image(img, f'C{klass}img{i}.png')
                
                # Trim extra examples
                examples[klass] = examples[klass][0:num_examples]

            self.gen.train()
            return examples

    def train(self, loader, learned_classes, model=None):
        gen_params = self.parameters['GEN_PARAMETERS']
        # Scale learning rate
        # gen_params['OPTIMIZER_PARAMETERS']['lr'] = gen_params['OPTIMIZER_PARAMETERS']['lr'] * len(learned_classes) / 10
        gen_opt = gen_params['OPTIMIZER'](self.gen.parameters(), **gen_params['OPTIMIZER_PARAMETERS'])
        # gen_scheduler = gen_params['SCHEDULER'](gen_opt, **gen_params['SCHEDULER_PARAMETERS'])
        
        discr_params = self.parameters['DISCR_PARAMETERS']
        discr_opt = discr_params['OPTIMIZER'](self.discr.parameters(), **discr_params['OPTIMIZER_PARAMETERS'])
        # discr_scheduler = discr_params['SCHEDULER'](discr_opt, **discr_params['SCHEDULER_PARAMETERS'])
        
        tensor = []
        g_vec = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            tensor.append(i)
        g_vec = g_vec.scatter_(1, torch.LongTensor(tensor).view(self.num_classes,1),
                               1).view(self.num_classes, self.num_classes, 1, 1)
        
        d_vec = torch.zeros([self.num_classes, self.num_classes, 32, 32])
        for i in range(self.num_classes):
			      d_vec[i, i, :, :] = 1

        print("Start training GAN on classes {}".format(learned_classes))

        for epoch in range(self.parameters['NUM_EPOCHS']):
            d_losses_e = []
            g_losses_e = []
            dist_losses_e = []

            self.gen.train()
            self.discr.train()

            # print("[GAN] Start Epoch {}\tGenerator LR:{}, Discriminator LR:{}".format(
            #     epoch + 1,
            #     gen_scheduler.get_last_lr(),
            #     discr_scheduler.get_last_lr()
            # ))

            start = time.time()

            for images, labels in loader:
                batch_size = images.shape[0]

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                # d_like_real = torch.ones(batch_size).to(DEVICE)
                # Label smoothing
                d_like_real = (torch.ones(batch_size) - 0.1).to(DEVICE)
                d_like_fake = torch.zeros(batch_size).to(DEVICE)

                ### Train Discriminator ###
                discr_opt.zero_grad()
                ## Train on real images
                d_labels = d_vec[labels]
                d_labels = d_labels.to(DEVICE)

                d_output_real = self.discr(images, d_labels).squeeze()

                ## Train on fake images
                # Random noise
                g_random_noise = torch.randn((batch_size, 100))
                g_random_noise = g_random_noise.view(-1, 100, 1, 1).to(DEVICE)
                #Generating random batch_size of labels from those present in activeClass
                random_labels = torch.from_numpy(np.random.choice(learned_classes, batch_size))
                #Convert labels to appropriate shapes
                g_random_labels = g_vec[random_labels].to(DEVICE)
                d_random_labels = d_vec[random_labels].to(DEVICE)
                #Generating fake images and passing them to discriminator
                g_output = self.gen(g_random_noise, g_random_labels)
                #Detach gradient from Generator
                g_output = g_output.detach()
                d_output_fake = self.discr(g_output, d_random_labels).squeeze()

                #Calculate BCE loss
                d_real_loss = self.criterion(d_output_real, d_like_real)
                d_fake_loss = self.criterion(d_output_fake, d_like_fake)
                d_loss = d_real_loss + d_fake_loss

                #Perform a backward step
                d_loss.backward()
                discr_opt.step()
                d_losses_e.append(d_loss.cpu().data.numpy())


                ### Train Generator ###
                gen_opt.zero_grad()

                g_output = self.gen(g_random_noise, g_random_labels)
                d_output = self.discr(g_output, d_random_labels).squeeze()

                g_loss = self.criterion(d_output, d_like_real)
                total_loss = g_loss

                # Lowers euclidean distance between features of generated and real images
                if model:
                    model.eval()
                    output_fake = model(g_output, get_only_features=True)
                    output_real = model(images, get_only_features=True)
                    distance_loss = torch.mean(nn.PairwiseDistance(2)(output_fake, output_real))
                    total_loss = g_loss + distance_loss
                    dist_losses_e.append(distance_loss.cpu().data.numpy())

                total_loss.backward()
                gen_opt.step()
                g_losses_e.append(g_loss.cpu().data.numpy())

            # gen_scheduler.step()
            # discr_scheduler.step()

            # Stats
            time_taken = time.time() - start
            mean_g = (sum(g_losses_e)/len(g_losses_e))
            mean_d = (sum(d_losses_e)/len(d_losses_e))
            mean_dist = (sum(dist_losses_e)/len(dist_losses_e))

            print("[GAN] Epoch: {}, g_loss: {}, d_loss: {}, dist_loss: {}, time taken: {}".format(
                      epoch + 1, mean_g, mean_d, mean_dist, time_taken
                  ))