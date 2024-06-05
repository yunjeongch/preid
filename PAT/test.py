import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from processor.part_attention_vit_processor import do_inference_with_save
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from utils.logger import setup_logger
from mmpose.apis import MMPoseInferencer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--save_dir", default='./inf/', help='path to save inference results', type = str
    )
    parser.add_argument(
        "--save", default=False, type = bool
    )
    parser.add_argument(
        "--pose", default=False, type = bool
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
    model.load_param(cfg.TEST.WEIGHT)

    save_dir = os.path.join(args.save_dir, cfg.LOG_NAME)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = save_dir + 'pred.jsonl'

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            if args.save:
                do_inference_with_save(cfg, model, val_loader, num_query, save_dir)
            elif args.pose:
                pose_model = MMPoseInferencer(pose2d=cfg.MODEL.MMPOSE_CONFIG, pose2d_weights=cfg.MODEL.MMPOSE_CKPT, device = "cuda")
                do_inf_pat(cfg, model, val_loader, num_query, pose_model)
            else:
                do_inf_pat(cfg, model, val_loader, num_query)
        else:
            do_inf(cfg, model, val_loader, num_query)
