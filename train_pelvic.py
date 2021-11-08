"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import numpy
import pdb


sys.path.append(os.path.join("..", "util"))
import common_metrics
import common_pelvic_pt as common_pelvic

def main(opts):
    # Load experiment setting
    config = get_config(opts.config)
    config["gpu"] = opts.gpu
    max_iter = config['max_iter']
    display_size = config['display_size']
    config['vgg_model_path'] = opts.output_path

    if opts.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
        cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Setup model and data loader
    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")

    if opts.gpu >= 0:
        trainer.cuda()

    device = trainer.s_a.device

    aug_para = {}
    if opts.aug_sigma:
        aug_para["sigma"] = opts.aug_sigma
        aug_para["points"] = opts.aug_points
        aug_para["rotate"] = opts.aug_rotate
        aug_para["zoom"] = opts.aug_zoom

    data_iter = common_pelvic.DataIter(device, opts.data_dir, patch_depth=config["input_dim_a"],
                                       batch_size=config["batch_size"])
    val_data_s, val_data_t, _, _ = common_pelvic.load_val_data(opts.data_dir)

    """
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)
    train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
    train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()
    """

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.log_dir, model_name))
    output_directory = os.path.join(opts.checkpoint_dir, "outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    for it in range(iterations, max_iter):
        trainer.update_learning_rate()
        images_a, images_b, _ = data_iter.next()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        """
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
        """

        if (iterations + 1) % config['image_display_iter'] == 0:
            msg = "Iter: %d" % (gen_iterations + 1)

            train_patch_s_np = patch_s.cpu().detach().numpy()
            train_patch_t_np = patch_t.cpu().detach().numpy()
            train_syn_st_np = syn_st.cpu().detach().numpy()
            train_syn_ts_np = syn_ts.cpu().detach().numpy()

            gen_images_train = numpy.concatenate([train_patch_s_np, train_syn_st_np, train_syn_ts_np, train_patch_t_np], 3)
            gen_images_train = common_pelvic.generate_display_image(gen_images_train, is_seg=False)

            if opts.log_dir:
                try:
                    skimage.io.imsave(os.path.join(opts.log_dir, "gen_images_train.jpg"), gen_images_train)
                except:
                    pass

            if args.do_validation:
                val_st_psnr = numpy.zeros((val_data_s.shape[0], 1), numpy.float32)
                val_ts_psnr = numpy.zeros((val_data_t.shape[0], 1), numpy.float32)
                val_st_list = []
                val_ts_list = []
                with torch.no_grad():
                    for i in range(val_data_s.shape[0]):
                        val_st = numpy.zeros(val_data_s.shape[1:], numpy.float32)
                        val_ts = numpy.zeros(val_data_t.shape[1:], numpy.float32)
                        used = numpy.zeros(val_data_s.shape[1:], numpy.float32)
                        for j in range(val_data_s.shape[1] - config["input_dim_a"] + 1):
                            patch_s = torch.tensor(val_data_s[i:i + 1, j:j + config["input_dim_a"], :, :], device=device)
                            patch_t = torch.tensor(val_data_t[i:i + 1, j:j + config["input_dim_b"], :, :], device=device)

                            ret_st, ret_ts = trainer.forward(patch_s, patch_t)

                            val_st[j:j + config["input_dim_a"], :, :] += ret_st
                            val_ts[j:j + config["input_dim_b"], :, :] += ret_ts
                            used[j:j + config["input_dim_b"], :, :] += 1

                    assert used.min() > 0
                    val_st /= used
                    val_ts /= used

                    st_psnr = common_metrics.psnr(val_st, val_data_t[i])
                    ts_psnr = common_metrics.psnr(val_ts, val_data_s[i])

                    val_st_psnr[i] = st_psnr
                    val_ts_psnr[i] = ts_psnr
                    val_st_list.append(st)
                    val_ts_list.append(ts)

                msg += "  val_st_psnr:%f/%f  val_ts_psnr:%f/%f" % \
                       (val_st_psnr.mean(), val_st_psnr.std(), val_ts_psnr.mean(), val_ts_psnr.std())
                gen_images_test = numpy.concatenate(
                    [self.val_data_s[0], val_st_list[0], val_ts_list[0], self.val_data_t[0]], 2)
                gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
                gen_images_test = common_pelvic.generate_display_image(gen_images_test, is_seg=False)

                if args.log_dir:
                    try:
                        skimage.io.imsave(os.path.join(args.log_dir, "gen_images_test.jpg"), gen_images_test)
                    except:
                        pass

                if val_ts_psnr.mean() > best_psnr:
                    best_psnr = val_ts_psnr.mean()

                    if best_psnr > args.psnr_threshold:
                        self.save_all_models(args.checkpoint_dir, "best")

                msg += "  best_ts_psnr:%f" % best_psnr

            self.logger.info(msg)


            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='configs/pelvic.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='outputs', help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default=r'data', help='path of the dataset')
    parser.add_argument('--log_dir', type=str, default=r'logs', help="log file dir")
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoints', help="checkpoint file dir")
    parser.add_argument('--pretrained_tag', type=str, default='best', choices=['best','final'], help="pretrained file tag")
    parser.add_argument('--aug_sigma', type=float, default=0, help="augmentation parameter:sigma")
    parser.add_argument('--aug_points', type=float, default=1, help="augmentation parameter:points")
    parser.add_argument('--aug_rotate', type=float, default=0, help="augmentation parameter:rotate")
    parser.add_argument('--aug_zoom', type=float, default=1., help="augmentation parameter:zoom")
    parser.add_argument('--psnr_threshold', type=float, default=18.0, help="only save the checkpoint file when PSNR reach this threshold")
    parser.add_argument('--do_validation', type=int, default=1, help="whether do validation during training")

    opts = parser.parse_args()

    main(opts)

