import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from collections import deque as collections_deque
from env_teacher import EnvTeacher

USE_INTRINSIC_REWARD = False
class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        self.env_teacher = EnvTeacher(self.env,
                                      args = {"ExploB_lmbd": 1.0,
                                        "ExploB_max": 1.0,
                                        "eta_phi_SelfRS": 0.01,
                                            "eta_phi_sors": 0.01,
                                            "eta_phi_rlirpg": 0.01,
                                            "eta_critic": 0.01,
                                            "sors_n_pairs": 10,
                                            "clipping_epsilon": 0.01,
                                            "use_clipping": False
                                        },
                                      teacher_name="SelfRS")

    def run(self):
        buffer = collections_deque(maxlen=self.trainer_params['buffer_size'])
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch, buffer)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, buffer):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # avg_score, avg_loss = self._generate_sampled_data(batch_size)
            epi_data = self._generate_sampled_data(batch_size)
            buffer.append(epi_data)
            if (loop_cnt + 1) % self.trainer_params['policy_update_freq'] == 0:
                avg_score, avg_loss = self._update_model(buffer)
            if (loop_cnt + 1) % self.trainer_params['reward_update_freq'] == 0:
                self._update_teacher(buffer)
            buffer.clear()
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            # if epoch == self.start_epoch:
            #     loop_cnt += 1
            #     if loop_cnt <= 10:
            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                        score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _generate_sampled_data(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem) i.e. tsp100: (64, 20, 0~100)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        epidata = []
        while not done:
            selected, probs, prob, state_embed = self.model(state)
            # from torchviz import make_dot
            # dot = make_dot(probs, params=dict(self.model.named_parameters()))
            # dot.render("conputation_graph", format="pdf")
            # shape: (batch, pomo)
            # state, reward, done = self.env.step(selected)
            state_embed = state_embed.clone().detach()
            next_state, reward_hat, done = self.env_teacher.step(selected, state_embed, done)

            e_t = {
                'state': state_embed.detach(),
                'action': selected.detach(),
                'reward_hat': reward_hat.detach(),
                'G_hat': None,
                'prob': prob.clone(),
                'probs': probs.clone().detach(),
                'done': done,
            }
            epidata.append(e_t)

            state = next_state
            # prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        G_hat = torch.zeros_like(epidata[-1]['reward_hat'])
        for i in range(len(epidata) - 1, -1, -1):
            reward = epidata[i]['reward_hat'].detach()
            G_hat = reward + self.env_teacher.gamma * G_hat
            epidata[i]['G_hat'] = G_hat
        return epidata

    def _update_model(self, buffer):
        recent_buffer_size = self.trainer_params['recent_buffer_size']
        loss_sum = 0
        score_sum = 0
        total_policy_loss = 0

        for episode_data in list(buffer)[-recent_buffer_size:]:
            states = torch.stack([step['state'] for step in episode_data])
            actions = torch.stack([step['action'] for step in episode_data])
            G_hats = torch.stack([step['G_hat'] for step in episode_data])
            prob = torch.stack([step['prob'] for step in episode_data])
            rewards = torch.stack([step['reward_hat'] for step in episode_data])
            
            # G_hats shape: (steps, batch, pomo)
            baseline = G_hats.mean(dim=2, keepdim=True)  # (steps, batch, 1)
            advantage = G_hats - baseline  # (steps, batch, pomo)
            
            log_probs = torch.log(prob + 1e-10)
            policy_loss = -(advantage * log_probs).mean()
            
            max_pomo_reward, _ = rewards[-1].max(dim=1)
            score = -max_pomo_reward.mean()
            
            total_policy_loss += policy_loss
            
            loss_sum += policy_loss.item()
            score_sum += score.item()

        avg_policy_loss = total_policy_loss / len(buffer)

        self.optimizer.zero_grad()
        avg_policy_loss.backward(retain_graph=True)
        self.optimizer.step()

        avg_loss = loss_sum / len(buffer)
        avg_score = score_sum / len(buffer)

        return avg_score, avg_loss
    
    def _update_teacher(self, buffer):
        return self.env_teacher.update(buffer)