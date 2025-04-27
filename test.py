import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDeblurDataset, RealworldTestDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F
from thop import profile
import matplotlib.pyplot as plt



class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()

    def forward(self,x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    psnr = AverageMeter()
    ssim = AverageMeter()

    selection = AverageMeter()
    t = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            start.record()
            restored = net(degrad_patch)
            end.record()

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            t.update(start.elapsed_time(end), N)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            selection.update(N)
            save_image_tensor(restored, output_path + clean_name[0] + '.png')

        flops, params = profile(net, inputs=(degrad_patch,), verbose=True)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')

        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))


'''
def visualize_prompt(prompt, save_path):
    """Visualizes a single prompt tensor."""
    # Assuming prompt shape: (batch_size, prompt_len, prompt_dim, H, W)
    prompt = prompt[0]  # Take the first prompt from the batch
    prompt = torch.mean(prompt, dim=0)  # Average across prompt length
    prompt = (prompt - prompt.min()) / (prompt.max() - prompt.min())  # Normalize to [0, 1]
    plt.imshow(prompt.cpu().numpy(), cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
'''


def test_Derain_Dehaze_Deblur(net, dataset, task="derain"):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')

        flops, params = profile(net, inputs=(degrad_patch,), verbose=True)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


def test_Realworld(net, dataset):
    output_path = testopt.output_path + 'realworld/'
    subprocess.check_output(['mkdir', '-p', output_path])

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    with torch.no_grad():
        for ([image_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.cuda()
            restored = net(degrad_patch)
            save_image_tensor(restored, os.path.join(output_path, image_name[0] + '.png'))

        # Since we don't have ground truth for real-world images, we can't calculate PSNR/SSIM
        flops, params = profile(net, inputs=(degrad_patch,), verbose=True)
        print('FLOPs = ' + str(flops/1000**3) + 'G')
        print('Params = ' + str(params/1000**2) + 'M')
        print("Real-world testing complete. Images saved to:", output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=2)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for deblur, 4 for all-in-one, 5 for realworld')

    parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--deblur_path', type=str, default="test/deblur", help='save path of test blurry images')
    parser.add_argument('--realworld_path', type=str, default="test/realworld/", help='path to realworld images')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="model.ckpt", help='checkpoint save path')
    testopt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name

    denoise_splits = ["urban100/"]
    derain_splits = ["Rain100L/"]
    deblur_splits = ["GoPro/"]

    denoise_tests = []
    derain_tests = []
    deblur_tests = []

    base_path = testopt.denoise_path
    for i in denoise_splits:
        testopt.denoise_path = os.path.join(base_path, i)
        denoise_testset = DenoiseTestDataset(testopt)
        denoise_tests.append(denoise_testset)

    print("CKPT name : {}".format(ckpt_path))

    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    if testopt.mode == 0:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

    elif testopt.mode == 1:
        print('Start testing rain streak removal...')
        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDeblurDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze_Deblur(net, derain_set, task="derain")

    elif testopt.mode == 2:
        print('Start testing dehazing images...')
        dehaze_base_path = testopt.dehaze_path
        #name = derain_splits[0]
        testopt.dehaze_path = os.path.join(dehaze_base_path)
        dehaze_set = DerainDehazeDeblurDataset(testopt, addnoise=False, sigma=15)
        test_Derain_Dehaze_Deblur(net, dehaze_set, task="dehaze")

    elif testopt.mode == 3:
        print('Start testing deblurring images...')
        deblur_base_path = testopt.deblur_path
        for name in deblur_splits:
            print('Start testing {} deblurring...'.format(name))
            testopt.deblur_path = os.path.join(deblur_base_path, name)
            deblur_set = DerainDehazeDeblurDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze_Deblur(net, deblur_set, task="deblur")

    elif testopt.mode == 4:
        for testset, name in zip(denoise_tests, denoise_splits):
            print('Start {} testing Sigma=15...'.format(name))
            test_Denoise(net, testset, sigma=15)

            print('Start {} testing Sigma=25...'.format(name))
            test_Denoise(net, testset, sigma=25)

            print('Start {} testing Sigma=50...'.format(name))
            test_Denoise(net, testset, sigma=50)

        derain_base_path = testopt.derain_path
        for name in derain_splits:
            print('Start testing {} rain streak removal...'.format(name))
            testopt.derain_path = os.path.join(derain_base_path, name)
            derain_set = DerainDehazeDeblurDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze_Deblur(net, derain_set, task="derain")

        print('Start testing dehazing images...')
        test_Derain_Dehaze_Deblur(net, derain_set, task="dehaze")

        print('Start testing deblurring images...')
        deblur_base_path = testopt.deblur_path
        for name in deblur_splits:
            testopt.deblur_path = os.path.join(deblur_base_path, name)
            deblur_set = DerainDehazeDeblurDataset(testopt, addnoise=False, sigma=15)
            test_Derain_Dehaze_Deblur(net, deblur_set, task="deblur")

    elif testopt.mode == 5:
        print('Start testing real-world images...')
        realworld_set = RealworldTestDataset(testopt)
        test_Realworld(net, realworld_set)


