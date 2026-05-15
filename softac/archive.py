from collections import namedtuple

import torch
from softac.modules import Actor, Critic

Batch = namedtuple("Batch", ["obs", "next_obs", "actions", "rewards", "dones"])


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
if __name__ == "__main__":

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=args.q_lr)
    alpha = log_alpha.exp().item()

    rb = TensorReplayBuffer(args.buffer_size, obs_dim, act_dim, device)

    # 3. Training Loop (Updates bound by Env Steps)
    obs, info = env.reset()
    num_updates = args.total_timesteps // args.n_envs

    for update in range(num_updates):
        global_step = update * args.n_envs

        with torch.no_grad():
            if global_step < args.learning_starts:
                actions = (
                    torch.rand((args.n_envs, act_dim), device=device) * 2 - 1
                ) * actor.scale + actor.bias
            else:
                actions, _, _ = actor.get_action(obs)

        next_obs, rewards, term, trunc, infos = env.step(actions)
        dones = term.float()  # Convert boolean/integer flags directly to float tensors

        rb.add(obs, next_obs, actions, rewards, dones)
        obs = next_obs

        # Update step (We sample and update just like a normal environment loop)
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_act, next_log_pi, _ = actor.get_action(data.next_obs)
                min_q = (
                    torch.min(
                        qf1_t(data.next_obs, next_act), qf2_t(data.next_obs, next_act)
                    )
                    - alpha * next_log_pi
                )
                target_q = data.rewards + (1 - data.dones) * args.gamma * min_q

            qf_loss = F.mse_loss(qf1(data.obs, data.actions), target_q) + F.mse_loss(
                qf2(data.obs, data.actions), target_q
            )
            q_opt.zero_grad()
            qf_loss.backward()
            q_opt.step()

            if update % args.policy_freq == 0:
                pi, log_pi, _ = actor.get_action(data.obs)
                actor_loss = (
                    (alpha * log_pi) - torch.min(qf1(data.obs, pi), qf2(data.obs, pi))
                ).mean()
                a_opt.zero_grad()
                actor_loss.backward()
                a_opt.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp().item()

            if update % args.target_network_freq == 0:
                for p, tp in zip(
                    list(qf1.parameters()) + list(qf2.parameters()),
                    list(qf1_t.parameters()) + list(qf2_t.parameters()),
                ):
                    tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

        if update % 50 == 0:
            print(f"Step: {global_step} / {args.total_timesteps} | Alpha: {alpha:.3f}")
