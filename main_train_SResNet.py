import os.path
import math
import argparse
import random
import numpy as np
import logging
import torch
from torch.utils.data import DataLoader


from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from dataset.dataset_SResNet import define_Dataset
from models.model_SResNet import Model_SResNet


'''
# --------------------------------------------
# training code for SResNet
# --------------------------------------------
# xiaoyang wang
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_SResNet.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.mkdirs(
        (path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(
        opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    border = 0
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(
        opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) /
                             dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True)
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = Model_SResNet(opt)

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1, 53):  # keep running
        for i, train_data in enumerate(train_loader):
            current_step += 1

            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            model.update_learning_rate()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                idx = 0
                info = ''

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    test_info_dir = opt['path']['test_info']
                    util.mkdir(img_dir)
                    util.mkdir(test_info_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()

                    L_img = util.tensor2uint(visuals['L'])
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image L and image E
                    # -----------------------
                    save_img_path = os.path.join(
                        img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    if current_step % opt['train']['checkpoint_save'] == 0:
                        util.imsave(E_img, save_img_path)
                    if current_step == opt['train']['checkpoint_test']:
                        noise_psnr = util.calculate_psnr(
                            L_img, H_img, border=border)
                        noise_ssim = util.calculate_ssim(
                            L_img, H_img, border=border)
                        info = '{:s}\n{:->4d}--> noise_picture | psnr:{:<4.2f}dB SSIM:{:<4.2f}'.format(
                            info, idx, noise_psnr, noise_ssim)
                        save_img_path = os.path.join(
                            img_dir, '{:s}_noise.png'.format(img_name))
                        util.imsave(L_img, save_img_path)

                    # -----------------------
                    # calculate PSNR and SSIM
                    # -----------------------
                    current_psnr = util.calculate_psnr(
                        E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(
                        E_img, H_img, border=border)

                    info = '{:s}\n{:->4d}--> {:>10s} | psnr:{:<4.2f}dB SSIM:{:<4.2f}'.format(
                        info, idx, image_name_ext, current_psnr, current_ssim)

                    avg_psnr += current_psnr
                    avg_ssim += current_ssim

                avg_psnr = avg_psnr/idx
                avg_ssim = avg_ssim/idx
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}'.format(
                    epoch, current_step, avg_psnr, avg_ssim))
                # save test_info
                info = '{:s}\niter:{:8,d}, Average PSNR : {:<.2f}dB, Average SSIM : {:<.4f}\n'.format(
                    info, current_step, avg_psnr, avg_ssim)
                save_info_path = os.path.join(
                    test_info_dir, '{:d}_info.txt'.format(current_step))
                f = open(save_info_path, 'w', encoding='utf-8')
                f.write(info)
                f.close()

            # -------------------------------
            # 6) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
