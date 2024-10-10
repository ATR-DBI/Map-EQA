import argparse
import torch
import os


def get_args():
    parser = argparse.ArgumentParser(
        description='Goal-Oriented-Semantic-Exploration')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=1)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('-n', '--num_processes', type=int, default=5,
                        help="""how many training processes to use (default:5)
                                Overridden when auto_gpu_config=1
                                and training on gpus""")
    parser.add_argument('--num_processes_per_gpu', type=int, default=6)
    parser.add_argument('--num_processes_on_first_gpu', type=int, default=1)
    parser.add_argument('--eval', type=int, default=0,
                        help='0: Train, 1: Evaluate (default: 0)')
    parser.add_argument('--num_training_frames', type=int, default=10000000,
                        help='total number of training frames')
    parser.add_argument('--num_eval_episodes', type=int, default=200,
                        help="number of test episodes per scene")
    parser.add_argument('--num_train_episodes', type=int, default=10000,
                        help="""number of train episodes per scene
                                before loading the next scene""")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=-1,
                        help="""gpu id for semantic model,
                                -1: same as sim gpu, -2: cpu""")

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="eqa",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='Model save frequency in number of updates')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length""")
    parser.add_argument("--task_config", type=str,
                        default="tasks/eqa_mp3d.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('--camera_height', type=float, default=1.25,
                        help="agent camera height in metres") # default: 0.88
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=30.0,
                        help="Maximum depth for depth sensor in meters") # before: 5.0, after: 20
    parser.add_argument('--success_dist', type=float, default=1.0,
                        help="success distance threshold in meters")
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor threshold in cm")
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")
    parser.add_argument('--version', type=str, default="v1",
                        help="dataset version")

    # Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='global_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--dist_coef', type=float, default=1.,
                        help='pred distance term coefficient (default: 0.01)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default="auto",
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")
    parser.add_argument('--reward_coeff', type=float, default=0.1,
                        help="Object goal reward coefficient")
    parser.add_argument('--intrinsic_rew_coeff', type=float, default=0.02,
                        help="intrinsic exploration reward coefficient")
    parser.add_argument('--num_sem_categories', type=int, default=8)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9,
                        help="Semantic prediction confidence threshold")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=2)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.20)

    # ADDED
    parser.add_argument('--with_info', type=bool, default=True)
    parser.add_argument('--sem_model', type=str, default="detectron2")
    # parser.add_argument('--depth_thresh', type=list, default=[1.0, 5.0])
    parser.add_argument('--use_baseline', type=list, default=0, help="Evaluation for baseline")
    parser.add_argument('--text_img_matching_threshold', type=float, default=0.01, help="Evaluation for baseline")
    parser.add_argument('--mask_dist', type=float, default=2.0,help="mask distance threshold in meters")
    parser.add_argument('--dividion_num', type=int, default=20,help="for allocating simulation thread to gpu")
    parser.add_argument('--dividion_num_first_gpu', type=int, default=20,help="for allocating simulation thread to the first gpu")
    
    parser.add_argument('--vqa_only', type=bool, default=False)
    parser.add_argument('--load_finetuned', type=bool, default=False)
    parser.add_argument('--vqa_chpt_path', type=str)
    parser.add_argument('--vqa_distance', type=float, default=1.0)
    parser.add_argument('--vqa_exp_name', type=str, default=None)
    parser.add_argument('--use_pano_for_itm', type=bool, default=False)
    parser.add_argument('--use_pano_for_vqa', type=bool, default=False)
    parser.add_argument('--forward_initial_goal_pred', type=bool, default=False)
    parser.add_argument('--back_step_num', type=str, default="rand_init", help="Set this values on [10,30,50, rand_init]")

    # For Detic
    # ADDED
    parser.add_argument('--detic_config_file', type=str, default="./libs/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument('--detic_checkpoint_file', type=str, default="./libs/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument('--detic_verbose', type=bool, default=True)
    parser.add_argument('--detic_vocabulary', type=str, default="custom_eqa")
    parser.add_argument('--detic_confidence_threshold', type=float, default=0.2) # TODO: set this value properly
    parser.add_argument("--semseg_shrink_disk_size", type=int, default=4)
    
    # For Global Policy
    parser.add_argument("--mask_nearest_locations", type=bool, default=True, help="")
    parser.add_argument("--nearest_frontier_mask_size", type=int, default=6)
    parser.add_argument("--move_as_close_as_possible", type=bool, default=True)
    parser.add_argument("--move_close_limit", type=int, default=25)
    parser.add_argument("--conseq_replan_thresh", type=int, default=5)

    # For Local Policy
    parser.add_argument("--stg_disk_size", type=int, default=10)
    parser.add_argument("--stg_downsampling", type=int, default=1)
    parser.add_argument("--weighted_scale", type=float, default=4.0)
    parser.add_argument("--weighted_niters", type=int, default=1)
    parser.add_argument("--stg_stop_dist", type=float, default=2.5)
    

    # Observation area for STG
    parser.add_argument("--obs_disk_size", type=int, default=6)

    # For vqa, saving images 
    # parser.add_argument("--dataset", type=str, default="mp3d")
    # save all images used on image-text nmatching
    parser.add_argument("--save_all_itm_images", type=bool, default=False)
    parser.add_argument("--dataset_version", type=str, default="v1")
    

    # parse arguments
    args = parser.parse_args()
    if "mp3d" in args.task_config:
        prefix_path = f"./data/datasets/eqa/mp3d/{args.dataset_version}/"
        args.episode_dir = prefix_path + "{split}/{split}.json.gz"
        args.content_dir = prefix_path + "{split}/content/"
        args.dataset = "mp3d"
    elif "scannet" in args.task_config:
        args.episode_dir = "./data/datasets/eqa/scannet/v1/{split}/{split}.json.gz"
        args.content_dir = "./data/datasets/eqa/scannet/v1/{split}/content/"
        args.dataset = "scannet"
    args.depth_thresh = [args.min_depth, args.max_depth -0.5] # TODO: set this value properly

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    if args.cuda:
        if args.auto_gpu_config:
            num_gpus = torch.cuda.device_count()
            if args.total_num_scenes != "auto":
                args.total_num_scenes = int(args.total_num_scenes)
            elif "eqa_mp3d" in args.task_config and \
                    "train" in args.split:
                args.total_num_scenes = 57
            elif "eqa_mp3d" in args.task_config and \
                    "val" in args.split:
                args.total_num_scenes = 10
            elif "eqa_scannet" in args.task_config:
                files = os.listdir(f"./data/datasets/eqa/scannet/v1/{args.split}/content/")
                args.total_num_scenes = len(files)
                print(f"TOTAL SCENE NUM: {args.total_num_scenes}")
            ####
            else:
                assert False, "Unknown task config, please specify" + \
                    " total_num_scenes"

            # GPU Memory required for the SemExp model:
            #       0.8 + 0.4 * args.total_num_scenes (GB)
            # GPU Memory required per thread: 2.6 (GB)
            if "eqa_mp3d" in args.task_config:
                min_memory_required = max(0.8 + 0.4 * args.total_num_scenes, 2.6)
            elif "eqa_scannet" in args.task_config:
                min_memory_required = 12 # Scannet EQA
            # min_memory_required = 2.6
            # Automatically configure number of training threads based on
            # number of GPUs available and GPU memory size
            gpu_memory = 1000
            for i in range(num_gpus):
                gpu_memory = min(gpu_memory,
                                 torch.cuda.get_device_properties(
                                     i).total_memory
                                 / 1024 / 1024 / 1024)
                assert gpu_memory > min_memory_required, \
                    """Insufficient GPU memory for GPU {}, gpu memory ({}GB)
                    needs to be greater than {}GB""".format(
                        i, gpu_memory, min_memory_required)

            # num_processes_per_gpu = int(gpu_memory / 2.6)
            num_processes_per_gpu = int(gpu_memory/ args.dividion_num)
            num_processes_on_first_gpu = \
                int((gpu_memory - min_memory_required) / args.dividion_num_first_gpu)
            # num_processes_on_first_gpu = 1

            if args.eval:
                max_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                # print(max_threads, args.total_num_scenes)
                # assert max_threads >= args.total_num_scenes, \
                #     """Insufficient GPU memory for evaluation"""

            if num_gpus == 1:
                args.num_processes_on_first_gpu = num_processes_on_first_gpu
                args.num_processes_per_gpu = 0
                args.num_processes = num_processes_on_first_gpu
                assert args.num_processes > 0, "Insufficient GPU memory"
            else:
                num_threads = num_processes_per_gpu * (num_gpus - 1) \
                    + num_processes_on_first_gpu
                num_threads = min(num_threads, args.total_num_scenes)
                args.num_processes_per_gpu = num_processes_per_gpu
                if args.vqa_only==True:
                    args.num_processes_on_first_gpu = 0
                else:    
                    args.num_processes_on_first_gpu = max(
                        0,
                        num_threads - args.num_processes_per_gpu * (num_gpus - 1))
                args.num_processes = num_threads
            # print(num_threads)
            args.sim_gpu_id = 1

            print("Auto GPU config:")
            print("Number of processes: {}".format(args.num_processes))
            print("Number of processes on GPU 0: {}".format(
                args.num_processes_on_first_gpu))
            print("Number of processes per GPU: {}".format(
                args.num_processes_per_gpu))
    else:
        args.sem_gpu_id = -2

    if args.num_mini_batch == "auto":
        args.num_mini_batch = max(args.num_processes // 2, 1)
    else:
        args.num_mini_batch = int(args.num_mini_batch)

    return args
