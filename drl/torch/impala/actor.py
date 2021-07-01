import queue
import shutil
from pathlib import Path
from typing import Union

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import gym
import utils
from learner import Learner
from models import MlpPolicy
from utils import Trajectory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class Actor(object):
    def __init__(
            self,
            id: int,
            hparams: utils.Hyperparameters,
            policy: MlpPolicy,
            learner: Learner,
            q: mp.Queue,
            update_counter: utils.Counter,
            log_path: Union[Path, str, None] = None,
            timeout=10,
    ):
        self.id = id
        self.hp = hparams
        self.policy = policy
        for p in self.policy.parameters():
            p.requires_grad = False
        self.learner = learner
        self.timeout = timeout
        self.q = q
        self.update_counter = update_counter
        self.log_path = log_path
        if self.log_path is not None:
            self.log_path = Path(self.log_path) / Path(f"a{self.id}")
            if self.log_path.exists():
                shutil.rmtree(self.log_path)
            self.log_path.mkdir(parents=True, exist_ok=False)

        self.completion = mp.Event()
        self.p = mp.Process(target=self._act, name="actor_{}".format(self.id))
        print("[main] actor_{} Initialized".format(self.id))

    def start(self):
        self.p.start()
        print("[main] Started actor_{} with pid {}".format(self.id, self.p.pid))

    def terminate(self):
        self.p.terminate()
        print("[main] Terminated actor_{}".format(self.id))

    def join(self):
        self.p.join()

    def _act(self):
        try:
            if self.log_path is not None:
                writer = SummaryWriter(self.log_path)
                writer.add_text("hyperparameters", "{}".format(self.hp))
            env = gym.make(self.hp.env_name)
            traj_no = 0

            while not self.learner.completion.is_set():
                traj_no += 1
                self.policy.load_state_dict(self.learner.policy_weights)
                traj = Trajectory(self.id, traj_no, [], [], [], [], [])
                obs = env.reset()
                obs = torch.tensor(obs, device=device, dtype=dtype)
                traj.obs.append(obs)
                c = 0
                if self.hp.verbose >= 2:
                    print("[actor_{}] Starting traj_{}".format(self.id, traj_no))

                # record trajectory
                while c < self.hp.max_timesteps:
                    if self.hp.render:
                        env.render()
                    c += 1
                    a, logits = self.policy.select_action(obs)
                    # print(f"[actor_{self.id}] a_probs: {a_probs}")
                    obs, r, done, _ = env.step(a.item())
                    obs = torch.tensor(obs, device=device, dtype=dtype)
                    r = torch.tensor(r, device=device, dtype=dtype)
                    done = torch.tensor(done, device=device)
                    traj.add(obs, a, r, done, logits)
                    if done:
                        break

                if self.hp.verbose >= 2:
                    print("[actor_{}] traj_{} completed Reward = {}".format(self.id, traj_no, traj.r))

                if self.log_path is not None:
                    writer.add_histogram("actor_{}/actions/action_taken".format(self.id), a, traj_no)
                    writer.add_histogram("actor_{}/actions/logits".format(self.id), logits.detach(), traj_no)
                    writer.add_scalar("actor_{}/rewards/trajectory_reward".format(self.id), sum(traj.r), traj_no, )

                while True:
                    try:
                        self.q.put(traj, timeout=self.timeout)
                        break
                    except queue.Full:
                        if self.learner.completion.is_set():
                            break
                        else:
                            continue

            if self.log_path is not None:
                writer.close()
            env.close()
            print("[actor_{}] Finished acting".format(self.id))
            self.completion.set()
            return
        except KeyboardInterrupt:
            print("[actor_{}] interrupted".format(self.id))
            if self.log_path is not None:
                writer.close()
            env.close()
            self.completion.set()
            return
        except Exception as e:
            if self.log_path is not None:
                writer.close()
            env.close()
            print("[actor_{}] encoutered exception".format(self.id))
            raise e
