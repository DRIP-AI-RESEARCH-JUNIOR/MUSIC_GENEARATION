import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *
from ops import *

def load_data():
  # TO do -: write this
  return train_loader, val_loader

def train(netG, netD, optimizerG, optimizerD, batch_size, nz, device=torch.device('cuda')):
  
  fixed_noise = torch.randn(batch_size, nz, device=device)
  netD.train()
  netG.train()
  
  real_label = 1
  fake_label = 0
  average_lossD = 0
  average_lossG = 0
  average_D_x   = 0
  average_D_G_z = 0

  lossD_list =  []
  lossD_list_all = []
  lossG_list =  []
  lossG_list_all = []
  D_x_list = []
  D_G_z_list = []
  
  for epoch in range(epochs):
    sum_lossD = 0
    sum_lossG = 0
    sum_D_x   = 0
    sum_D_G_z = 0
    
    for i, (data,prev_data,chord) in enumerate(train_loader, 0):
      
      #############################################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      #############################################################
      
      # train with real      
      netD.zero_grad()
      real_cpu = data.to(device)
      prev_data_cpu = prev_data.to(device)
      chord_cpu = chord.to(device)
      
      batch_size = real_cpu.size(0)
      label = torch.full((batch_size,), real_label, device=device)
      D, D_logits, fm = netD(real_cpu,chord_cpu,batch_size,pitch_range)
      
      #####loss
      d_loss_real = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, 0.9*torch.ones_like(D)))
      d_loss_real.backward(retain_graph=True)
      D_x = D.mean().item()
      sum_D_x += D_x 
      
      # train with fake
      noise = torch.randn(batch_size, nz, device=device)
      fake = netG(noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
      label.fill_(fake_label)
      D_, D_logits_, fm_ = netD(fake.detach(),chord_cpu,batch_size,pitch_range)
      d_loss_fake = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.zeros_like(D_)))
      
      d_loss_fake.backward(retain_graph=True)
      D_G_z1 = D_.mean().item()
      errD = d_loss_real + d_loss_fake
      errD = errD.item()
      lossD_list_all.append(errD)
      sum_lossD += errD
      optimizerD.step()
      
      #############################################
      # (2) Update G network: maximize log(D(G(z)))
      #############################################
      
      netG.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost
      D_, D_logits_, fm_= netD(fake,chord_cpu,batch_size,pitch_range)
      
      ###loss
      g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
      #Feature Matching
      features_from_g = reduce_mean_0(fm_)
      features_from_i = reduce_mean_0(fm)
      fm_g_loss1 =torch.mul(l2_loss(features_from_g, features_from_i), 0.1)

      mean_image_from_g = reduce_mean_0(fake)
      smean_image_from_i = reduce_mean_0(real_cpu)
      fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), 0.01)

      errG = g_loss0 + fm_g_loss1 + fm_g_loss2
      errG.backward(retain_graph=True)
      D_G_z2 = D_.mean().item()
      optimizerG.step()
      
      ###################################################
      # (3) Update G network again: maximize log(D(G(z)))
      ###################################################
      
      netG.zero_grad()
      label.fill_(real_label)  # fake labels are real for generator cost
      D_, D_logits_, fm_ = netD(fake,chord_cpu,batch_size,pitch_range)

      ###loss
      g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
      #Feature Matching
      features_from_g = reduce_mean_0(fm_)
      features_from_i = reduce_mean_0(fm)
      loss_ = nn.MSELoss(reduction='sum')
      feature_l2_loss = loss_(features_from_g, features_from_i)/2
      fm_g_loss1 =torch.mul(feature_l2_loss, 0.1)

      mean_image_from_g = reduce_mean_0(fake)
      smean_image_from_i = reduce_mean_0(real_cpu)
      mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i)/2
      fm_g_loss2 = torch.mul(mean_l2_loss, 0.01)
      errG = g_loss0 + fm_g_loss1 + fm_g_loss2
      sum_lossG +=errG
      errG.backward()
      lossG_list_all.append(errG.item())

      D_G_z2 = D_.mean().item()
      sum_D_G_z += D_G_z2
      optimizerG.step()
      
      if i % 100 == 0:
        vutils.save_image(real_cpu, '%s/real_samples.png' % 'file', normalize=True)
        fake = netG(fixed_noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % ('file', epoch), normalize=True)
  
  average_lossD = (sum_lossD / len(train_loader))
  average_lossG = (sum_lossG / len(train_loader))
  average_D_x = (sum_D_x / len(train_loader))
  average_D_G_z = (sum_D_G_z / len(train_loader))
  
  lossD_list.append(average_lossD)
  lossG_list.append(average_lossG)
  D_x_list.append(average_D_x)
  D_G_z_list.append(average_D_G_z)
  
  print('==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} '.format(
    epoch, average_lossD,average_lossG,average_D_x, average_D_G_z))
  
  # Save plot
  length = lossG_list.shape[0]
  x = np.linspace(0, length-1, length)
  x = np.asarray(x)
  
  plt.figure()
  plt.plot(x, lossD_list,label=' lossD',linewidth=1.5)
  plt.plot(x, lossG_list,label=' lossG',linewidth=1.5)
  
  plt.savefig('where you want to save/lr='+ str(lr) +'_epoch='+str(epochs)+'.png')
  
  
  
