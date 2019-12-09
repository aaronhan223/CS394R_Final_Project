import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
#from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.spmodel import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

from tensorboardX import SummaryWriter

def check_path(path):
    if not os.path.exists(path):
        print("[INFO] making folder %s" % path)
        os.makedirs(path)

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy},
        dimh=args.dimh)
    actor_critic.to(device)

    exp_name = "%s_%s_seed%d_dimh%d_" % (args.env_name, args.algo, args.seed, args.dimh)
    if args.gail:
        exp_name += '_gail_'

    if args.split:
        exp_name += 'splitevery' + str(args.split_every)
        if args.random_split:
            exp_name += '_rsplit'
    else:
        exp_name += 'baseline'

    writer = SummaryWriter('./runs/'+exp_name)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    print(num_updates)
    stats = {
            'seed': args.seed,
            'experiment': exp_name,
            'env': args.env_name,
            'dimh': args.dimh,
            'split every': args.split_every,
            'random split': args.random_split,
            'steps': [],
            'mean reward': [],
            'actor neurons': [],
            'critic neurons': [],
    }
    save_dir = './experiment_results/%s/' % args.env_name
    stats_save_path = save_dir + exp_name
    check_path(save_dir)
    print('start')
    count = -1
    num_updates = 488 * 2
    meanreward = []
    for j in range(num_updates):
        #if j % 50 == 0:
        #    print('STEP', j)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            count += 1
            if j % 488 == 0:
                count = 0
                total = 488 * 2
            else:
                total = 488 * 2
            if args.split:
                utils.update_linear_schedule(
                    agent.optimizer, count, total,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)
            else:
                utils.update_linear_schedule(
                            agent.optimizer, j, num_updates,
                                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        # splitting
        if args.split and (j+1) % args.split_every == 0 and j < 200:
            print("[INFO] split on iteration %d..." % j)
            agent.split(rollouts, args.random_split)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))
        meanreward.append(np.mean(episode_rewards))
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            if True:
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))
            stats['mean reward'].append(np.mean(episode_rewards))
            stats['steps'].append(j)
            if args.split:
                a, c = agent.actor_critic.get_num_params()
                stats['actor neurons'].append(a)
                stats['critic neurons'].append(c)

        if j % 10 == 0:
            print("[INFO] saving to ", stats_save_path)
            np.save(stats_save_path, stats)

        
        if j % 5 == 0:
            s = (j + 1) * args.num_processes * args.num_steps
            if args.split:
                a, c = agent.actor_critic.get_num_params()
                writer.add_scalar('A neurons', a, s)
                writer.add_scalar('C neurons', c, s)
            writer.add_scalar('mean reward', np.mean(episode_rewards) ,s)
            writer.add_scalar('entropy loss', dist_entropy ,s)
            writer.add_scalar('value loss', value_loss ,s)
            writer.add_scalar('action loss', action_loss ,s)
        

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

    writer.close()
    import pickle
    pickle.dump(meanreward, open(stats_save_path + '.pkl', 'wb'))


if __name__ == "__main__":
    main()
