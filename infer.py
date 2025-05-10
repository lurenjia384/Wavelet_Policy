"""
Author: Changchuan Yang
Email: ccyang@zjut.edu.cn
"""
import argparse
import os
import numpy as np
import torch
import logging
import datetime
from pytorch_wavelets import DWT1DInverse
import numpy as np
import pickle
import threading
import matplotlib.pyplot as plt
from utils import make_sim_env, BOX_POSE, sample_box0_pose, sample_box1_pose, sample_box_pose,\
      sample_insertion_pose, sample_box2_pose, cosine_similarity, save_videos, get_image

tt=1

class Test:
    def __init__(self, opt, net=None):
        self.opt = opt
        self.iscuda = torch.cuda.is_available()
        self.device = f'cuda:{torch.cuda.current_device()}' if self.iscuda and not opt.cpu else 'cpu'
        self.net = net
        self.model_path = None
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.MAE = torch.nn.L1Loss(reduction='mean')
        self.online_model_list = {"sim_transfer_cube_scripted":"task_1/best_model_" + str(opt.cam_num) + "1.pt", 
                                    "sim_insertion_scripted":"task_2/best_model_" + str(opt.cam_num) + "2.pt",
                                    "sim_transfer_cube_scripted_plus":"task_3/best_model_" + str(opt.cam_num) + "3.pt",
                                    "Put":"task_4/best_model_" + str(opt.cam_num) + "4.pt"}
        self.online_data_stats_list = {"sim_transfer_cube_scripted":"task_1/task_1.pkl", 
                                    "sim_insertion_scripted":"task_2/task_2.pkl",
                                    "sim_transfer_cube_scripted_plus":"task_3/task_3.pkl",
                                    "Put":"task_4/task_4.pkl"}
        if net is not None:
            assert next(self.net.parameters()).device == self.device

    def load_model(self, model_path):
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id="lurenjia384/wavelet_policy_model", 
                filename=self.online_model_list[opt.task_name]
                )
        self.model_path = model_path
        self.net = torch.load(model_path, map_location=self.device)
        print(f"Loaded model from {model_path}\n")
        total_params = sum(p.numel() for p in self.net.parameters())
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6
        print(f"Total number of model parameters: {total_params_m:.2f}M, trainable parameter quantity: {trainable_params_m:.2f}M")
        self.net.eval()

    def test(self):
        print_str = f'model details:\n{self.net.args}'
        self.log_path = self.opt.logdir + f'/{datetime.datetime.now().strftime("%y-%m-%d")}'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if opt.netdir:
            file_name = os.path.basename(opt.netdir)
        else:
            file_name = "best_model_13.pt"
        
        logging.basicConfig(filename=os.path.join(self.log_path, f'test_{self.net.args["train_opt"].comments}{file_name}.log'),
                            filemode='w', format='%(asctime)s   %(message)s', level=logging.DEBUG)
        logging.debug(print_str)
        logging.debug(self.model_path)
        camera_names = ["top", 'angle', 'vis']  # camera name order
        camera_names = camera_names[0:opt.cam_num]
        task_name = opt.task_name
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward
        num_rollouts = 50
        episode_returns = []
        highest_rewards = []
        onscreen_cam = 'angle'
        num_success = 0
        error = 0
        change = None
        for rollout_id in range(num_rollouts):
            rollout_id += 0
            if tt==0:
                ax = plt.subplot()
                plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
                plt.ion()
            
            if opt.netdir is None:
                from huggingface_hub import hf_hub_download
                stats_path = hf_hub_download(
                    repo_id="lurenjia384/wavelet_policy_model", 
                    filename= self.online_data_stats_list[opt.task_name]
                    )
            else :
                stats_path = opt.stats_path
            with open(stats_path,  'rb') as f:
                stats = pickle.load(f)
            pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
            post_process = lambda a: a * stats['action_std'] + stats['action_mean']
            last_action=[]
            rewards = []
            image_list=[]
            with torch.inference_mode():
                if 'sim_transfer_cube_scripted_plus' in task_name:
                    BOX_POSE[0] = np.concatenate([sample_box0_pose(), sample_box1_pose()]) 
                elif 'sim_transfer_cube' in task_name:
                    BOX_POSE[0] = sample_box_pose() 
                elif 'sim_insertion' in task_name:
                    BOX_POSE[0] = np.concatenate(sample_insertion_pose()) 
                elif 'Put' in task_name:
                    BOX_POSE[0] = np.concatenate([sample_box0_pose(), sample_box1_pose(), sample_box2_pose()])
                ts = env.reset()
                warn = 0
                for t in range(600):
                    obs = ts.observation
                    if tt==0:
                        image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                        plt_img.set_data(image)
                        plt.pause(0.002)
                    start_ts=t
                    if 'images' in obs:
                        image_list.append(obs['images'])
                    else:
                        image_list.append({'main': obs['image']})
                    icurr_image = get_image(ts, camera_names)
                    idwt = DWT1DInverse(wave=self.net.args['train_opt'].wavelet, mode=self.net.args['train_opt'].wt_mode).cuda()
                    img=icurr_image.to('cuda')
                    qpos_numpy = np.array(obs['qpos'])
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    wt_pre_batch, wt_pre_batch_oth, _= self.net(qpos, img, [], None)
                    pre_batch = idwt((wt_pre_batch[-1].transpose(1, 2).contiguous(),
                                        [comp.transpose(1, 2).contiguous() for comp in
                                        wt_pre_batch[:-1]])).contiguous() 
                    pre_batch = pre_batch.transpose(1, 2) 
                    result=pre_batch
                    if result.shape[-1] == 7:
                        pre_batch_oth = idwt((wt_pre_batch_oth[-1].transpose(1, 2).contiguous(),
                                            [comp.transpose(1, 2).contiguous() for comp in
                                            wt_pre_batch_oth[:-1]])).contiguous()
                        pre_batch_oth = pre_batch_oth.transpose(1, 2)
                        result_oth=pre_batch_oth
                        result = torch.cat((result[:, :-1, :], 
                        result_oth[:, :-1, :], 
                        ), 
                        dim=-1).to(result.device)
                    raw_action = result[:,20+0-2:20+10+0-2,:].cpu().numpy()
                    action = post_process(raw_action)
                    raw_action = action
                    if len(last_action) < 10:
                        last_action.append(raw_action)
                    else :
                        last_action.pop(0)
                        last_action.append(raw_action)
                    weights = torch.zeros(10, dtype=torch.float, device='cpu')
                    weights[1]=0
                    weights[0]=1.0
                    if warn != 0 :
                        weights = torch.zeros(10, dtype=torch.float, device='cpu')
                        weights[int(warn/2)]=1.0
                        warn = warn - 1
                    weights_np = weights.cpu().numpy().reshape(1, -1) 
                    weighted_sum = np.matmul(weights_np, raw_action)
                    weighted_sum = np.squeeze(weighted_sum, axis=1)
                    weighted_sum = np.array(weighted_sum)
                    target_qpos = weighted_sum
                    if change is None:
                        change = weighted_sum
                    else :
                        print(f"{start_ts} Cosine Similarity: {(1-cosine_similarity(change[0:6], weighted_sum[0:6]))*1000000:.8f}", 
                                100*np.linalg.norm(change[0:14] - weighted_sum[0:14]), end='\r')
                        if 1000*np.linalg.norm(change[0:6] - weighted_sum[0:6]) + \
                            1000*np.linalg.norm(change[7:13] - weighted_sum[7:13])< opt.stop_time:
                            warn = 4
                            print("warning!", end='\r')
                        else :
                            if warn !=0:
                                warn = warn - 1
                            change = weighted_sum
                    target_qpos = target_qpos.squeeze(0)
                    ts = env.step(target_qpos)
                    rewards.append(ts.reward)
                if tt==0:
                    plt.close()

                save_videos(image_list, 0.02, video_path=os.path.join("./vid_path", f'{file_name}video{rollout_id}.mp4'))
                rewards = np.array(rewards)
                episode_return = np.sum(rewards[rewards!=None])
                episode_returns.append(episode_return)
                episode_highest_reward = np.max(rewards)
                highest_rewards.append(episode_highest_reward)
                logging.debug(f'Rollout {rollout_id}\nepisode_return={episode_return}, episode_highest_reward={episode_highest_reward}, env_max_reward={env_max_reward}, Success: {episode_highest_reward == env_max_reward}')
                print(f'Rollout {rollout_id}\nepisode_return={episode_return}, episode_highest_reward={episode_highest_reward}, env_max_reward={env_max_reward}, Success: {episode_highest_reward == env_max_reward}')
                if episode_highest_reward == env_max_reward:
                    num_success = num_success + 1
                    print(file_name,"success:", num_success)
                    logging.debug(f"{file_name} success: {num_success}")
                else :
                    error = error + 1
                    if error > opt.exit_num :
                        exit()

if __name__ == '__main__':
    # torch.cuda.set_per_process_memory_fraction(0.022)
    # torch.cuda.set_per_process_memory_fraction(0.5) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--logdir', default='./log', type=str)
    parser.add_argument('--netdir', default=None, type=str)
    parser.add_argument('--no_visualization', default=1, type=int)
    parser.add_argument('--cam_num', default=1, type=int)
    parser.add_argument('--stop_time', default=0.4, type=float, help="This value is recommended to be 3.5 in the task of insertion if the robotic arm is stuck or if there are idle segments in the dataset.")
    parser.add_argument('--exit_num', default= 50, type=int)
    parser.add_argument('--task_name', default="sim_transfer_cube_scripted_plus", type=str)
    parser.add_argument('--stats_path', default="dataset_stats.pkl", type=str)
    opt = parser.parse_args()
    tt = opt.no_visualization
    def listen_input():
        global tt 
        while True:
            user_input = input()
            if user_input == "q":
                break
            elif user_input == "0":
                tt=0
            elif user_input == "1":
                tt=1
            else:
                print("Invalid command, please re-enter")
    thread = threading.Thread(target=listen_input, daemon=True)
    thread.start()
    test = Test(opt)
    test.load_model(opt.netdir)
    test.test()
