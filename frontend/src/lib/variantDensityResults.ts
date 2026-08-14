export interface DensityResult {
  win_rate: number;
  avg_episode_length: number;
  avg_reward: number;
}

/**
 * Real evaluation results (2,000 greedy episodes, seed 42) for each DQN/PPO
 * ablation variant's own checkpoint, at beginner/sparse (3 mines) and
 * beginner/dense (8 mines) on the same 5x5 board it was trained on --
 * beginner/standard (5 mines) isn't repeated here since that's exactly what
 * the run's own `RunBrief`/`ExperimentDetail.evaluation_metrics` already is.
 * Regenerate with `python -m evaluation.rescore_variant_densities --episodes 2000`,
 * which prints this block ready to paste.
 *
 * Keyed by the real experiment id (`RunBrief.id`), and covering exactly the
 * runs the research pipeline renders as cards. The previous contents keyed the
 * five DQN variants since retired to `HIDDEN_VARIANTS` plus four PPO runs
 * (`ppo_exp_A_baseline`...`ppo_exp_D_shaped_checkpoint`) no longer shipped in
 * `results_public/` at all -- so every row described a run the site never
 * renders, while all fourteen visible runs fell through to no data.
 *
 * Each variant is scored on the checkpoint it actually deployed, resolved from
 * its own summary's `used_checkpoint`, and under the board distribution its own
 * summary records in `eval_env`. Both matter for the same reason: these two
 * cells sit in one table beside the run's headline Standard number, so all
 * three have to describe the same weights playing the same game. Scoring a
 * different checkpoint file, or scoring a `first_click_safe: area` run on the
 * v1 default, would read as a density effect when it is a deployment or
 * distribution one.
 */
export const VARIANT_DENSITY_RESULTS: Record<string, { sparse: DensityResult; dense: DensityResult }> = {
  exp_F_masked_target: {
    sparse: { win_rate: 0.254, avg_episode_length: 4.6815, avg_reward: -1.2385 },
    dense: { win_rate: 0.002, avg_episode_length: 3.405, avg_reward: -7.555 },
  },
  exp_G_masked_lr_decay: {
    sparse: { win_rate: 0.295, avg_episode_length: 4.9385, avg_reward: -0.1615 },
    dense: { win_rate: 0.0005, avg_episode_length: 3.298, avg_reward: -7.692 },
  },
  exp_H_masked_shaped: {
    sparse: { win_rate: 0.326, avg_episode_length: 5.143, avg_reward: 0.663 },
    dense: { win_rate: 0.001, avg_episode_length: 3.2395, avg_reward: -7.7405 },
  },
  exp_I_masked_slow_epsilon: {
    sparse: { win_rate: 0.274, avg_episode_length: 4.7355, avg_reward: -0.7845 },
    dense: { win_rate: 0.004, avg_episode_length: 3.4585, avg_reward: -7.4615 },
  },
  exp_J_masked_deep: {
    sparse: { win_rate: 0.193, avg_episode_length: 4.7615, avg_reward: -2.3785 },
    dense: { win_rate: 0.001, avg_episode_length: 3.3435, avg_reward: -7.6365 },
  },
  exp_K_masked_longer: {
    sparse: { win_rate: 0.371, avg_episode_length: 5.1615, avg_reward: 1.5815 },
    dense: { win_rate: 0.005, avg_episode_length: 3.182, avg_reward: -7.718 },
  },
  exp_L_tuned: {
    sparse: { win_rate: 0.6165, avg_episode_length: 5.3925, avg_reward: 6.7225 },
    dense: { win_rate: 0.0065, avg_episode_length: 3.428, avg_reward: -7.442 },
  },
  exp_N_no_reward_scale: {
    sparse: { win_rate: 0.584, avg_episode_length: 5.5215, avg_reward: 6.2015 },
    dense: { win_rate: 0.008, avg_episode_length: 3.4845, avg_reward: -7.3555 },
  },
  exp_O_short_epsilon: {
    sparse: { win_rate: 0.472, avg_episode_length: 5.37, avg_reward: 3.81 },
    dense: { win_rate: 0.005, avg_episode_length: 3.237, avg_reward: -7.663 },
  },
  exp_P_train_every_1: {
    sparse: { win_rate: 0.2495, avg_episode_length: 4.7285, avg_reward: -1.2815 },
    dense: { win_rate: 0.0005, avg_episode_length: 3.0415, avg_reward: -7.9485 },
  },
  exp_M_fully_conv: {
    sparse: { win_rate: 0.7035, avg_episode_length: 5.9515, avg_reward: 9.0215 },
    dense: { win_rate: 0.033, avg_episode_length: 3.808, avg_reward: -6.532 },
  },
  ppo_exp_E_longer_matched: {
    sparse: { win_rate: 0.097, avg_episode_length: 4.058, avg_reward: -5.002 },
    dense: { win_rate: 0.0005, avg_episode_length: 2.88, avg_reward: -8.11 },
  },
  ppo_exp_F_shaped_matched: {
    sparse: { win_rate: 0.12, avg_episode_length: 3.6245, avg_reward: -4.9755 },
    dense: { win_rate: 0.0005, avg_episode_length: 2.846, avg_reward: -8.144 },
  },
  ppo_exp_G_shaped_ckpt_matched: {
    sparse: { win_rate: 0.159, avg_episode_length: 3.8555, avg_reward: -3.9645 },
    dense: { win_rate: 0.0, avg_episode_length: 2.764, avg_reward: -8.236 },
  },
  ppo_long_A_baseline: {
    sparse: { win_rate: 0.109, avg_episode_length: 4.0415, avg_reward: -4.7785 },
    dense: { win_rate: 0.0005, avg_episode_length: 2.962, avg_reward: -8.028 },
  },
  ppo_long_B_shaped: {
    sparse: { win_rate: 0.1205, avg_episode_length: 3.8745, avg_reward: -4.7155 },
    dense: { win_rate: 0.0, avg_episode_length: 2.8435, avg_reward: -8.1565 },
  },
  ppo_long_D_gamma09_matched: {
    sparse: { win_rate: 0.1235, avg_episode_length: 4.241, avg_reward: -4.289 },
    dense: { win_rate: 0.001, avg_episode_length: 2.91, avg_reward: -8.07 },
  },
  ppo_long_E_fully_conv: {
    sparse: { win_rate: 0.1855, avg_episode_length: 3.8635, avg_reward: -3.4265 },
    dense: { win_rate: 0.0, avg_episode_length: 2.8005, avg_reward: -8.1995 },
  },
  ppo_long_F_shaped_reseeded: {
    sparse: { win_rate: 0.125, avg_episode_length: 3.725, avg_reward: -4.775 },
    dense: { win_rate: 0.0, avg_episode_length: 2.807, avg_reward: -8.193 },
  },
  dqn_v2_A_baseline: {
    sparse: { win_rate: 0.894, avg_episode_length: 3.644, avg_reward: 10.524 },
    dense: { win_rate: 0.389, avg_episode_length: 5.0995, avg_reward: 1.8795 },
  },
  ppo_v2_F_shaped_matched: {
    sparse: { win_rate: 0.3305, avg_episode_length: 2.8955, avg_reward: -1.4945 },
    dense: { win_rate: 0.0135, avg_episode_length: 2.8665, avg_reward: -7.8635 },
  },
  ppo_v2_G_solvable_matched: {
    sparse: { win_rate: 0.1625, avg_episode_length: 2.48, avg_reward: -5.27 },
    dense: { win_rate: 0.009, avg_episode_length: 2.9585, avg_reward: -7.8615 },
  },
  ppo_v2_H_gamma09_matched: {
    sparse: { win_rate: 0.156, avg_episode_length: 3.9345, avg_reward: -3.9455 },
    dense: { win_rate: 0.0015, avg_episode_length: 3.3605, avg_reward: -7.6095 },
  },
  ppo_v2_I_gamma09_solvable_matched: {
    sparse: { win_rate: 0.147, avg_episode_length: 3.96, avg_reward: -4.1 },
    dense: { win_rate: 0.0045, avg_episode_length: 3.094, avg_reward: -7.816 },
  },
  ppo_v2_J_fully_conv: {
    sparse: { win_rate: 0.2985, avg_episode_length: 2.8495, avg_reward: -2.1805 },
    dense: { win_rate: 0.0095, avg_episode_length: 2.9795, avg_reward: -7.8305 },
  },
  ppo_v2_K_solvable_fully_conv: {
    sparse: { win_rate: 0.1875, avg_episode_length: 3.335, avg_reward: -3.915 },
    dense: { win_rate: 0.014, avg_episode_length: 3.2325, avg_reward: -7.4875 },
  },
};
