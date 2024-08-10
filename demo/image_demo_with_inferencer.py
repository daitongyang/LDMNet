# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import MMSegInferencer

#
# python E:\daity\mmsegmentation-at-af\mmsegmentation\demo\image_demo_with_inferencer.py E:\daity\dataset\MaSTr1325\images\0463.jpg E:\daity\mmsegmentation-at-af\mmsegmentation\configs\ddrnet\ltbanet-s-mastr1325.py --checkpoint E:\daity\mmsegmentation-at-af\mmsegmentation\checkpoints\MaSTr1325\LTbANet-s-9623-MaSTr1325-300k-512×512iter_213000.pth


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('model', help='Config file')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='', help='Path to save result file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='Whether to display the drawn image.')
    parser.add_argument(
        '--dataset-name',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    mmseg_inferencer = MMSegInferencer(
        args.model,
        args.checkpoint,
        classes=(
            'Static Obstacle', 'Water', 'Sky','',''),
        palette=[[247, 195, 37], [41, 167, 224], [90, 75, 164], [247, 195, 37], [247, 195, 37]],
        # dataset_name=args.dataset_name,
        device=args.device)

    # test a single image
    mmseg_inferencer(
        args.img,show=args.show, out_dir=args.out_dir, opacity=args.opacity)


if __name__ == '__main__':
    main()
