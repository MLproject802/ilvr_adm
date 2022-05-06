import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision.transforms as transforms
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.image_datasets import load_data
from torchvision import utils
from resizer import Resizer
import math


# added
def load_reference(data_dir, batch_size, image_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,
        deterministic=True,
        random_flip=False,
    )
    for source_img, target_img, source_inst, target_inst, model_kwargs in data:
        model_kwargs["ref_img"] = source_img
        model_kwargs["ref_img_target"] = target_img
        model_kwargs["inst"] = source_inst
        model_kwargs["inst_target"] = target_inst
        yield model_kwargs


def main():
    args = create_argparser().parse_args()

    # th.manual_seed(0)

    dist_util.setup_dist()
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.down_N, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs = next(data)
        # labellist=[2,3,4,5,10,11,12,13]
        labellist=[10]
        source_inst_map=model_kwargs['inst'][0]
        target_inst_map=model_kwargs['inst_target'][0]
        source=model_kwargs['ref_img'][0]
        # print("aaaaaaaaaaaaa",source.shape)
        inst_squeeze=source_inst_map
        inst_squeeze_target=target_inst_map
        source_test=th.zeros([3,256,256]).cuda()
        for label in labellist:
            # print("aaaaaaaaaaaaaaa",inst_squeeze.shape)
            flag=(inst_squeeze==label)
            flag_target=(inst_squeeze_target==label)
            flag_list=th.nonzero(flag,as_tuple=False).float()
            # print(flag_list)
            flag_list_target=th.nonzero(flag_target,as_tuple=False).float()
            if(len(flag_list)!=0 and len(flag_list_target)!=0):
                center_area_source_pos=th.mean(flag_list,dim=0)
                center_area_target_pos=th.mean(flag_list_target,dim=0)
                move_dist=center_area_target_pos-center_area_source_pos
                # print("======",move_dist)
                # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                # print("center_area_pos",move_dist)
                for i in range(3):
                    if(move_dist[0]<=0):
                        # print("aaa")
                        # print("aaaaaaaaa",source_inst_map.shape)
                        source_inst_narrow=source_inst_map.narrow(0,int(-move_dist[0]),256-int(-move_dist[0])).cuda()
                        concate=th.zeros([int(-move_dist[0]),256]).cuda()
                        source_inst_moved_dim0=th.cat([source_inst_narrow,concate],dim=0).cuda()
                        source_narrow=source[i].narrow(0,int(-move_dist[0]),256-int(-move_dist[0])).cuda()
                        concate=th.zeros([int(-move_dist[0]),256]).cuda()
                        source_moved_dim0=th.cat([source_narrow,concate],dim=0).cuda()
                    #print("cocate_size",source_inst_moved_dim0.shape)
                    else:
                        source_inst_narrow=source_inst_map.narrow(0,0,256-int(move_dist[0])).cuda()
                        concate=th.zeros([int(move_dist[0]),256]).cuda()
                        source_inst_moved_dim0=th.cat([concate,source_inst_narrow],dim=0).cuda()
                        source_narrow=source[i].narrow(0,0,256-int(move_dist[0])).cuda()
                        concate=th.zeros([int(move_dist[0]),256]).cuda()
                        source_moved_dim0=th.cat([concate,source_narrow],dim=0).cuda()
                    if(move_dist[1]<=0):
                        source_inst_narrow_dim1=source_inst_moved_dim0.narrow(1,int(-move_dist[1]),256-int(-move_dist[1])).cuda()
                        concate=th.zeros([256,int(-move_dist[1])]).cuda()
                        source_inst_moved=th.cat([source_inst_narrow_dim1,concate],dim=1).cuda()
                        source_narrow_dim1=source_moved_dim0.narrow(1,int(-move_dist[1]),256-int(-move_dist[1])).cuda()
                        concate=th.zeros([256,int(-move_dist[1])]).cuda()
                        source_moved=th.cat([source_narrow_dim1,concate],dim=1).cuda()
                    else:
                        source_inst_narrow_dim1=source_inst_moved_dim0.narrow(1,0,256-int(move_dist[1])).cuda()
                        concate=th.zeros([256,int(move_dist[1])]).cuda()
                        source_inst_moved=th.cat([concate,source_inst_narrow_dim1],dim=1).cuda()
                        source_narrow_dim1=source_moved_dim0.narrow(1,0,256-int(move_dist[1])).cuda()
                        concate=th.zeros([256,int(move_dist[1])]).cuda()
                        source_moved=th.cat([concate,source_narrow_dim1],dim=1).cuda()
                    # print("============",th.sum(source_inst_moved))
                    source_id_area=th.where((source_inst_moved>=2)*(source_inst_moved<=13) ,1,0).cuda()
                    source_moved=th.where(source_id_area.byte(),source_moved,source_id_area.float())
                    source_test[i]+=source_moved
                    model_kwargs['ref_img_target'][0][i]=th.where(source_id_area.byte().cpu(),source_moved.cpu(),model_kwargs['ref_img_target'][0][i])
                
        # print(model_kwargs['ref_img_target'].shape)
        # unloader=transforms.ToPILImage()
        # image = source_test.cpu().clone()
        # image = image.squeeze(0)
        # image = unloader(image)
        # image_path = "./output/source_test.png"
        # image.save(image_path)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=args.range_t
        )

        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=1,
        down_N=8,
        range_t=0,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()