#!/usr/bin/env python3
import functools
import os

from baselines import logger
from mpi4py import MPI
import mpi_util
import tf_util
from cmd_util import make_atari_env, arg_parser
from policies.cnn_gru_policy_dynamics import CnnGruPolicy
from policies.cnn_policy_param_matched import CnnPolicy
from ppo_agent import PpoAgent
from utils import set_global_seeds
from vec_env import VecFrameStack

# Profiling libraries
# import cProfile, tracemalloc

def train(*, env_id, num_env, hps, num_timesteps, seed):
    experiment = os.environ.get('EXPERIMENT_LVL')
    if experiment == 'ego':
        # for the ego experiment we needed a higher intrinsic coefficient
        hps['int_coeff'] = 3.0

    logger.info("Hyperparameters:")
    logger.info(hps)
    venv = VecFrameStack(
        make_atari_env(env_id, num_env, seed, wrapper_kwargs={},
                       start_index=num_env * MPI.COMM_WORLD.Get_rank(),
                       max_episode_steps=hps.pop('max_episode_steps')),
        hps.pop('frame_stack'))
    venv.score_multiple = {'Mario': 500,
                           'MontezumaRevengeNoFrameskip-v4': 1,
                           'GravitarNoFrameskip-v4': 250,
                           'PrivateEyeNoFrameskip-v4': 500,
                           'SolarisNoFrameskip-v4': None,
                           'VentureNoFrameskip-v4': 200,
                           'PitfallNoFrameskip-v4': 100,
                           }[env_id]

    venv.record_obs = True if env_id == 'SolarisNoFrameskip-v4' else False
    ob_space = venv.observation_space
    ac_space = venv.action_space

    gamma = hps.pop('gamma')
    policy = {'rnn': CnnGruPolicy,
              'cnn': CnnPolicy}[hps.pop('policy')]

    agent = PpoAgent(
        scope='ppo',
        ob_space=ob_space,
        ac_space=ac_space,
        stochpol_fn=functools.partial(
            policy,
                scope='pol',
                ob_space=ob_space,
                ac_space=ac_space,
                update_ob_stats_independently_per_gpu=hps.pop('update_ob_stats_independently_per_gpu'),
                proportion_of_exp_used_for_predictor_update=hps.pop('proportion_of_exp_used_for_predictor_update'),
                dynamics_bonus = hps.pop("dynamics_bonus")
            ),
        gamma=gamma,
        gamma_ext=hps.pop('gamma_ext'),
        lam=hps.pop('lam'),
        nepochs=hps.pop('nepochs'),
        nminibatches=hps.pop('nminibatches'),
        lr=hps.pop('lr'),
        cliprange=0.1,
        nsteps=128,
        ent_coef=0.001,
        max_grad_norm=hps.pop('max_grad_norm'),
        use_news=hps.pop("use_news"),
        comm=MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None,
        update_ob_stats_every_step=hps.pop('update_ob_stats_every_step'),
        int_coeff=hps.pop('int_coeff'),
        ext_coeff=hps.pop('ext_coeff'),
        restore_model=hps.pop('restore_model')
    )
    agent.start_interaction([venv])
    if hps.pop('update_ob_stats_from_random_agent'):
        agent.collect_random_statistics(num_timesteps=128*50)

    save_model = hps.pop('save_model')
    assert len(hps) == 0, "Unused hyperparameters: %s" % list(hps.keys())

    #profiler = cProfile.Profile()
    #profiler.enable()
    #tracemalloc.start()
    #prev_snap = tracemalloc.take_snapshot()
    counter = 0
    while True:
        info = agent.step()
        if info['update']:
            logger.logkvs(info['update'])
            logger.dumpkvs()
            counter += 1

            # if (counter % 10) == 0:
            #     snapshot = tracemalloc.take_snapshot()
            #     top_stats = snapshot.compare_to(prev_snap, 'lineno')
            #     for stat in top_stats[:10]:
            #         print(stat)
            #     prev_snap = snapshot
            #     profiler.dump_stats("profile_rnd")

            if (counter % 100) == 0 and save_model:
                agent.save_model(agent.I.step_count)

        if agent.I.stats['tcount'] > num_timesteps:
            break

    agent.stop_interaction()


def add_env_params(parser):
    parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4',
                        choices=['MontezumaRevengeNoFrameskip-v4', 'GravitarNoFrameskip-v4',
                                 'VentureNoFrameskip-v4', 'PitfallNoFrameskip-v4'])
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--max_episode_steps', type=int, default=4500)


def main():
    parser = arg_parser()
    add_env_params(parser)
    parser.add_argument('--num-timesteps', type=int, default=int(2.01e8))
    parser.add_argument('--num_env', type=int, default=32)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gamma_ext', type=float, default=0.999)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--update_ob_stats_every_step', type=int, default=0)
    parser.add_argument('--update_ob_stats_independently_per_gpu', type=int, default=0)
    parser.add_argument('--update_ob_stats_from_random_agent', type=int, default=1)
    parser.add_argument('--proportion_of_exp_used_for_predictor_update', type=float, default=1.)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--policy', type=str, default='cnn', choices=['cnn', 'rnn'])
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--ext_coeff', type=float, default=2.)
    parser.add_argument('--dynamics_bonus', type=int, default=0)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--restore_model', action='store_true')
    parser.add_argument('--experiment', type=str, default='ego', choices=['baseline', 'attention', 'ego'])

    args = parser.parse_args()
    logger.configure(dir=logger.get_dir(), format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else [])
    if MPI.COMM_WORLD.Get_rank() == 0:
        with open(os.path.join(logger.get_dir(), 'experiment_tag.txt'), 'w') as f:
            f.write(args.tag)
        # shutil.copytree(os.path.dirname(os.path.abspath(__file__)), os.path.join(logger.get_dir(), 'code'))

    mpi_util.setup_mpi_gpus()

    seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)

    os.environ["EXPERIMENT_LVL"] = args.experiment
    hps = dict(
        frame_stack=4,
        nminibatches=4,
        nepochs=4,
        lr=0.0001,
        max_grad_norm=0.0,
        use_news=args.use_news,
        gamma=args.gamma,
        gamma_ext=args.gamma_ext,
        max_episode_steps=args.max_episode_steps,
        lam=args.lam,
        update_ob_stats_every_step=args.update_ob_stats_every_step,
        update_ob_stats_independently_per_gpu=args.update_ob_stats_independently_per_gpu,
        update_ob_stats_from_random_agent=args.update_ob_stats_from_random_agent,
        proportion_of_exp_used_for_predictor_update=args.proportion_of_exp_used_for_predictor_update,
        policy=args.policy,
        int_coeff=args.int_coeff,
        ext_coeff=args.ext_coeff,
        dynamics_bonus = args.dynamics_bonus,
        save_model=args.save_model,
        restore_model=args.restore_model
    )

    tf_util.make_session(make_default=True)
    train(env_id=args.env, num_env=args.num_env, seed=seed,
        num_timesteps=args.num_timesteps, hps=hps)


if __name__ == '__main__':
    main()
