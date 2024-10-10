# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# this code is based on a file in home-robot repository


import argparse
import pathlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


from centernet.config import add_centernet_config  # noqa: E402


sys.path.insert(0, 'libs/Detic/')
from detic.config import (  # noqa: E402
    add_detic_config,
)
from detic.modeling.text.text_encoder import (  # noqa: E402
    build_text_encoder,
)
from detic.modeling.utils import (  # noqa: E402
    reset_cls_test,
)

BUILDIN_CLASSIFIER = {
    "lvis": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}

detic_category_to_task_category_id = {
    "chair": 0,
    "table": 1,
    "picture": 2,
    "cabinet": 3,
    "cushion": 4,
    "sofa": 5,
    "bed": 6,
    "chest_of_drawers": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
    "stool": 11,
    "towel": 12,
    "tv_monitor": 13,
    "shower": 14,
    "bathtub": 15,
    "counter": 16,
    "fireplace": 17,
    "gym_equipment": 18,
    "seating": 19,
    "clothes": 20,
    "curtain": 21,
    "door": 22,
    "shelving": 23,
    "oven": 24,
    "microwave": 25,
    "stove": 26,
    "wardrobe": 27,
    'refrigerator': 28,
    "blinds": 29,
    "washing_machine": 30,
    "clothes_dryer": 31,
    "fridge": 28,
    "tv_stand": 32,
}

mapping_thda_to_objectnav = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25:24,
    26:25,
    27:26,
    28:27,
    29:28,
    30:29,
    31:30,
    32:31,
    33:32
}
BACKGROUND = 100

def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

def overlay_masks(
    masks: np.ndarray, class_idcs: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Overlays the masks of objects
    Determines the order of masks based on mask size
    """
    mask_sizes = [np.sum(mask) for mask in masks]
    sorted_mask_idcs = np.argsort(mask_sizes)

    semantic_mask = np.zeros(shape)
    instance_mask = -np.ones(shape)
    for i_mask in sorted_mask_idcs[::-1]:  # largest to smallest
        semantic_mask[masks[i_mask].astype(bool)] = class_idcs[i_mask]
        instance_mask[masks[i_mask].astype(bool)] = i_mask

    return semantic_mask, instance_mask


class SemanticPredDetic:
    """
    Directly take a batch of RGB images (tensors) and get batched predictions.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # Convert model to device
        sem_gpu_id = self.cfg.sem_gpu_id
        # Setup model
        self.model = DeticPerception(
            config_file=cfg.detic_config_file,
            vocabulary=cfg.detic_vocabulary,
            custom_vocabulary="",
            checkpoint_file=cfg.detic_checkpoint_file,
            sem_gpu_id=cfg.sem_gpu_id,
            verbose=cfg.detic_verbose,
            confidence_threshold=cfg.detic_confidence_threshold
        )

        if sem_gpu_id == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{sem_gpu_id}")
        # Normalization
        self.input_max_depth = self.cfg.max_depth
        self.input_min_depth = self.cfg.min_depth



    def get_predictions(self, batched_rgb, batched_depth):
        """
        Inputs:
            batched_rgb - (B, 3, H, W) RGB float Tensor values in [0.0, 255.0]
            batched_depth - (B, 1, H, W) depth float Tensor values in meters
        Outputs: (B, N, H, W) segmentation masks
        """
        _, _, H, W = batched_rgb.shape
        raw_depth = batched_depth
        self.rgb_with_bbox = []
        with torch.no_grad():
            batched_rgb = batched_rgb.permute(0, 2, 3, 1) # convert rgb shape to (B,H,W,3)
            predictions_semantic = []
            predictions_instance = []
            for i in range(0, batched_rgb.shape[0]):
                rgb = batched_rgb[i]
                rgb_ndarray = rgb.to('cpu').detach().numpy().copy()
                pred = self.model.predict(rgb=rgb_ndarray, draw_instance_predictions=True)
                self.rgb_with_bbox.append(pred["task_observations"]["semantic_frame"])
                self.pred_semantic = pred["semantic"]
                if self.model.num_sem_categories==1:
                    # pred["instance"] is larger than -1. pixels on pred["instance"]==-1 mean that they are background. we have to exclude that area.
                    pred_semantic_wo_background = np.where(pred["instance"]>=0, 0, -1) # convert background to -1
                else: # TODO: not correct. We do not use this one
                    pred_semantic_wo_background = np.where(pred["instance"]>=0, pred["semantic"], -1) # convert background to -1
                # print(f"Instance IDs: {np.unique(pred['instance'])}, Semantics: {np.unique(pred['semantic'])}")
                predictions_semantic.append(pred_semantic_wo_background)
                predictions_instance.append(pred["instance"])
                torch.cuda.empty_cache()
        predictions_semantic = np.array(predictions_semantic)
        
        semantic_inputs = self.process_predictions(predictions_semantic, raw_depth)
        return semantic_inputs

    def process_predictions(self, predictions, raw_depth):
        B, H, W = predictions.shape
        semantic_inputs = torch.zeros(
            B, self.cfg.num_sem_categories, H, W, device=self.device
        )
        predictions = torch.from_numpy(predictions.astype(np.float32)).clone().to(self.device)

        # Ignore predictions for pixels that are outside the depth threshold
        is_within_thresh = ((raw_depth[:, 0] >= self.cfg.depth_thresh[0]) & (
            raw_depth[:, 0] <= self.cfg.depth_thresh[1]
        )).to(self.device)

        for mapping_key, class_idx in self.model.categories_mapping.items():
            if class_idx >= self.cfg.num_sem_categories:
                continue
            semantic_inputs[:, class_idx] = (predictions == mapping_key) * is_within_thresh
            semantic_inputs[:, class_idx] = (predictions == mapping_key)

        return semantic_inputs


class DeticPerception:
    def __init__(
        self,
        config_file=None,
        vocabulary="coco",
        custom_vocabulary="",
        checkpoint_file=None,
        sem_gpu_id=0,
        verbose: bool = False,
        confidence_threshold: float = 0.45,
    ):
        """Load trained Detic model for inference.

        Arguments:
            config_file: path to model config
            vocabulary: currently one of "coco" for indoor coco categories or "custom"
             for a custom set of categories
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        self.verbose = verbose
        if config_file is None:
            config_file = str(
                Path(__file__).resolve().parent
                / "Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
            )
        if checkpoint_file is None:
            checkpoint_file = str(
                Path(__file__).resolve().parent
                / "Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
            )
        if self.verbose:
            print(
                f"Loading Detic with config={config_file} and checkpoint={checkpoint_file}"
            )

        string_args = f"""
            --config-file {config_file} --vocabulary {vocabulary}
            """

        if vocabulary == "custom":
            assert custom_vocabulary != ""
            string_args += f""" --custom_vocabulary {custom_vocabulary}"""

        string_args += f""" --opts MODEL.WEIGHTS {checkpoint_file}"""

        if sem_gpu_id == -1:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""

        string_args = string_args.split()
        args = get_parser().parse_args(string_args)
        args.confidence_threshold = confidence_threshold
        cfg = setup_cfg(args, verbose=verbose)

        assert vocabulary in ["coco", "custom", "custom_eqa", "zeroshot_eqa"]
        if args.vocabulary == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(",")
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
        elif args.vocabulary == "custom_eqa": # For EQA object categories
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = [
                'cabinet', 'picture', 'sofa', 'curtain', 'door', 'shelving',
                'chest_of_drawers', 'table', 'fireplace', 'oven', 'shower',
                'cushion', 'microwave', 'stove', 'counter', 'plant', 'wardrobe',
                'bed', 'refrigerator', 'towel', 'blinds', 'washing_machine',
                'clothes_dryer', 'stool', 'chair', 'fridge', 'clothes',
                'seating', 'sink', 'tv_stand'
            ]
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {
                i: detic_category_to_task_category_id[cat] for i,cat in enumerate(self.metadata.thing_classes)
            }
            self.categories_mapping[BACKGROUND] = BACKGROUND
            # self.categories_mapping = {
            #     i: i for i in range(len(self.metadata.thing_classes))
            # }
        elif args.vocabulary == "coco":
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]
            self.categories_mapping = {
                56: 0,  # chair
                57: 1,  # couch
                58: 2,  # plant
                59: 3,  # bed
                61: 4,  # toilet
                62: 5,  # tv
                60: 6,  # table
                69: 7,  # oven
                71: 8,  # sink
                72: 9,  # refrigerator
                73: 10,  # book
                74: 11,  # clock
                75: 12,  # vase
                41: 13,  # cup
                39: 14,  # bottle
            }

        self.num_sem_categories = len(self.categories_mapping)

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE
        cfg.defrost()
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
            '/home/dbi-data7/k.sakamoto/Documents/Object-Goal-Navigation/libs/Detic/datasets/metadata/lvis_v1_train_cat_info.json'
        cfg.freeze()
        # import pdb;pdb.pdb.set_trace()
        self.predictor = DefaultPredictor(cfg)

        if type(classifier) == pathlib.PosixPath:
            classifier = str(classifier)
        reset_cls_test(self.predictor.model, classifier, num_classes)
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
    
    def get_text_embeddings(self, vocabulary, prompt="a "):
        texts = [prompt + x for x in vocabulary]
        emb = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return emb

    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        if self.verbose:
            print(f"Resetting vocabulary to {new_vocab}")
        MetadataCatalog.remove("__unused")
        if vocab_type == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = new_vocab
            # classifier = get_clip_embeddings(self.metadata.thing_classes)
            classifier = self.get_text_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
        else:
            raise NotImplementedError(
                "Detic does not have support for resetting from custom to coco vocab"
            )
        self.num_sem_categories = len(self.categories_mapping)

        num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def predict(
        self,
        rgb,
        draw_instance_predictions: bool = True,
    ):
        """
        Arguments:
            rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR) of ndarray
            depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        # import pdb;pdb.pdb.set_trace()
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape

        pred = self.predictor(image)
        pred_result = {}

        task_observations = {}
        if draw_instance_predictions:
            visualizer = Visualizer(
                image[:, :, ::-1], self.metadata, instance_mode=self.instance_mode
            )
            visualization = visualizer.draw_instance_predictions(
                predictions=pred["instances"].to(self.cpu_device)
            ).get_image()
            task_observations["semantic_frame"] = visualization
        else:
            task_observations["semantic_frame"] = None

        # Sort instances by mask size
        masks = pred["instances"].pred_masks.cpu().numpy()
        class_idcs = pred["instances"].pred_classes.cpu().numpy()
        scores = pred["instances"].scores.cpu().numpy()

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        semantic = semantic_map.astype(int)
        instance = instance_map.astype(int)
        if task_observations is None:
            task_observations = dict()
        task_observations["instance_map"] = instance_map
        task_observations["instance_classes"] = class_idcs
        task_observations["instance_scores"] = scores

        pred_result["task_observations"] = task_observations
        pred_result["semantic"] = semantic
        pred_result["instance"] = instance
        self.pred_result = pred_result
        return pred_result

def setup_cfg(args, verbose: bool = False):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    if verbose:
        print("[DETIC] Confidence threshold =", args.confidence_threshold)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    # Fix cfg paths given we're not running from the Detic folder
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(
        Path(__file__).resolve().parent / "Detic" / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom","custom_eqa"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
