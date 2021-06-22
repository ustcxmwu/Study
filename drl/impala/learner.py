import queue
import shutil
from pathlib import Path
from typing import Union

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import utils
from models import MlpPolicy, MlpValueFn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


class Learner(object):
    def __init__(
            self,
            id: int,
            hparams: utils.Hyperparameters,
            policy: MlpPolicy,
            value_fn: MlpValueFn,
            q: mp.Queue,
            update_counter: utils.Counter,
            log_path: Union[str, Path, None] = None,
            timeout=200,
    ):
        self.id = id
        self.hp = hparams
        self.policy = policy
        self.value_fn = value_fn
        self.optimizer = torch.optim.Adam(
            [*self.policy.parameters(), *self.value_fn.parameters()], lr=self.hp.lr
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 0.95)
        self.timeout = timeout
        self.q = q
        self.update_counter = update_counter
        self.log_path = log_path
        if self.log_path is not None:
            self.log_path = Path(log_path) / Path("l{}".format(self.id))
            if self.log_path.exists():
                shutil.rmtree(self.log_path)
            self.log_path.mkdir(parents=True, exist_ok=False)

        self.completion = mp.Event()
        self.p = mp.Process(target=self._learn, name="learner_{}".format(self.id))
        print("[main] learner_{} Initialized".format(self.id))

    def start(self):
        self.completion.clear()
        self.p.start()
        print("[main] Started learner_{} with pid {}".format(self.id, self.p.pid))

    def terminate(self):
        self.p.terminate()
        print("[main] Terminated learner_{}".format(self.id))

    def join(self):
        self.p.join()

    def _learn(self):
        try:
            update_count = 0

            if self.log_path is not None:
                writer = SummaryWriter(self.log_path)
                writer.add_text("hyperparameters", "{}".format(self.hp))

            while update_count < self.hp.max_updates:
                if self.hp.verbose >= 2:
                    print("[learner_{}] Beginning Update_{}".format(self.id, update_count))

                # set up tracking variables
                traj_count = 0
                value_fn_loss = 0.0
                policy_loss = 0.0
                policy_entropy = 0.0
                loss = torch.zeros(1, device=device, dtype=dtype, requires_grad=True)
                reward = 0.0

                # process batch of trajectories
                while traj_count < self.hp.batch_size:
                    try:
                        traj = self.q.get(timeout=self.timeout)
                    except queue.Empty as e:
                        print("[learner_{}] No trajectory recieved for {} seconds, Exiting.".format(self.id,
                                                                                                    self.timeout))
                        if self.log_path is not None:
                            writer.close()
                        self.completion.set()
                        raise e

                    if self.hp.verbose >= 2:
                        print("[learner_{}] Processing traj_{}".format(self.id, traj.id))
                    traj_len = len(traj.r)
                    obs = torch.stack(traj.obs)
                    actions = torch.stack(traj.a)
                    r = torch.stack(traj.r)
                    reward += torch.sum(r).item() / self.hp.batch_size
                    disc = self.hp.gamma * (~torch.stack(traj.d))

                    # compute value estimates and logits for observed states
                    v = self.value_fn(obs).squeeze(1)
                    curr_logits = self.policy(obs[:-1])

                    # compute log probs for current and old policies
                    curr_log_probs = action_log_probs(curr_logits, actions)
                    traj_log_probs = action_log_probs(torch.stack(traj.logits), actions)

                    # computing v trace targets recursively
                    with torch.no_grad():
                        imp_sampling = torch.exp(
                            curr_log_probs - traj_log_probs
                        ).squeeze(1)
                        rho = torch.clamp(imp_sampling, max=self.hp.rho_bar)
                        c = torch.clamp(imp_sampling, max=self.hp.c_bar)
                        delta = rho * (r + self.hp.gamma * v[1:] - v[:1])
                        vt = torch.zeros(traj_len + 1, device=device, dtype=dtype)

                        for i in range(traj_len - 1, -1, -1):
                            vt[i] = delta[i] + disc[i] * c[i] * (vt[i + 1] - v[i + 1])
                        vt = torch.add(vt, v)

                        # vt = (vt - torch.mean(vt)) / torch.std(vt)

                        pg_adv = rho * (r + disc * vt[1:] - v[:-1])

                    # print(f"v: {v}")
                    # print(f"vt: {vt}")
                    # print(f"pg_adv: {pg_adv}")
                    # print(f"rho: {rho}")

                    # compute loss as sum of value loss, policy loss and entropy
                    # traj_value_fn_loss = 0.5 * torch.sum(torch.pow(v - vt, 2))
                    # traj_policy_loss = torch.sum(curr_log_probs * pg_adv.detach())
                    # traj_policy_entropy = -1 * torch.sum(
                    #     F.softmax(curr_logits, dim=-1)
                    #     * F.log_softmax(curr_logits, dim=-1)
                    # )
                    traj_value_fn_loss = compute_baseline_loss(v - vt)
                    traj_policy_loss = compute_policy_gradient_loss(
                        curr_logits, actions, pg_adv
                    )
                    traj_policy_entropy = -1 * compute_entropy_loss(curr_logits)
                    traj_loss = (
                            self.hp.v_loss_c * traj_value_fn_loss
                            + self.hp.policy_loss_c * traj_policy_loss
                            - self.hp.entropy_c * traj_policy_entropy
                    )
                    loss = torch.add(loss, traj_loss / self.hp.batch_size)
                    value_fn_loss += traj_value_fn_loss.item() / self.hp.batch_size
                    policy_loss += traj_policy_loss.item() / self.hp.batch_size
                    policy_entropy += traj_policy_entropy.item() / self.hp.batch_size
                    traj_count += 1

                if self.hp.verbose >= 2:
                    print("[learner_{}] Updating model weights for Update {}".format(self.id, update_count))

                # backpropogating loss and updating weights
                # self.policy_optimizer.zero_grad()
                # self.value_fn_optimizer.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.hp.max_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_fn.parameters(), self.hp.max_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                # self.policy_optimizer.step()
                # self.value_fn_optimizer.step()

                # log to console
                if self.hp.verbose >= 1:
                    print("[learner_{}] Update {} | Batch Mean Reward: {:.2f} | Loss: {:.2f}".format(self.id,
                                                                                                     update_count,
                                                                                                     reward,
                                                                                                     loss.item()))

                # evaluate current policy
                if self.hp.eval_every is not None:
                    if (update_count + 1) % self.hp.eval_every == 0:
                        eval_r, eval_std = utils.evaluate(
                            self.policy,
                            self.hp.env_name,
                            self.hp.eval_eps,
                            True,
                            self.hp.max_timesteps,
                        )
                        if self.hp.verbose >= 1:
                            print(
                                "[learner_{}] Update {} | Evaluation Reward: {:.2f}, Std Dev: {:.2f}".format(self.id,
                                                                                                             update_count,
                                                                                                             eval_r,
                                                                                                             eval_std))
                        if self.log_path is not None:
                            writer.add_scalar("learner_{}/rewards/evaluation_reward".format(self.id), eval_r,
                                              update_count + 1, )

                # log to tensorboard
                if self.log_path is not None:
                    writer.add_scalar("learner_{}/rewards/batch_mean_reward".format(self.id), reward,
                                      update_count + 1, )
                    writer.add_scalar("learner_{}/loss/policy_loss".format(self.id), policy_loss, update_count + 1, )
                    writer.add_scalar("learner_{}/loss/value_fn_loss".format(self.id), value_fn_loss,
                                      update_count + 1, )
                    writer.add_scalar("learner_{}/loss/policy_entropy".format(self.id), policy_entropy,
                                      update_count + 1, )
                    writer.add_scalar("learner_{}/loss/total_loss".format(self.id), loss, update_count + 1)

                # save model weights every given interval
                if (update_count + 1) % self.hp.save_every == 0:
                    path = self.log_path / Path("IMPALA_{}_l{}_{}.pt".format(env.name, self.id, update_count))
                    self.save(path)
                    print("[learner_{}] Saved model weights at update {} to {}".format(self.id, update_count, path))

                # increment update counter
                self.update_counter.increment()
                update_count = self.update_counter.value

            if self.log_path is not None:
                writer.close()

            print("[learner_{}] Finished learning".format(self.id))
            self.completion.set()
            return

        except KeyboardInterrupt:
            print("[learner_{}] Interrupted".format(self.id))
            if self.log_path is not None:
                writer.close()
            self.completion.set()
            return

        except Exception as e:
            if self.log_path is not None:
                writer.close()
            print("[learner_{}] Encoutered exception".format(self.id))
            raise e

    def save(self, path):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "value_fn_state_dict": self.value_fn.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_fn.load_state_dict(checkpoint["value_fn_state_dict"])

    @property
    def policy_weights(self) -> torch.Tensor:
        return self.policy.state_dict()


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(policy_logits, dim=-1),
        target=torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(logits, dim=-1), target=torch.flatten(actions), reduction="none",
    ).view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())
