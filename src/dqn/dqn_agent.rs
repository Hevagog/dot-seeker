use crate::core::agent::{ActionSpace, PlayerAction};
use bevy::prelude::*;

use crate::dqn::dqn_net::*;

#[derive(Resource, Debug)]
pub struct DQNAgent {
    observation_space: usize,
    action_space: ActionSpace,
    episodes: usize,
    pub batch_size: usize,
    learning_rate: f64,
    pub gamma: f64,
    eps_start: f64,
    eps_end: f64,
    eps_decay: usize,
    replay_buffer: Vec<(Vec<f64>, usize, f64, Vec<f64>, bool)>,
    steps_done: usize,
    // dqn_model: DQNModel<burn::backend::DefaultBackend>,
}

impl DQNAgent {
    pub fn new(observation_space: usize, action_space: ActionSpace) -> Self {
        let action_size = action_space.n_discrete().unwrap_or(0);
        if action_size == 0 {
            panic!("Action space must have at least one action.");
        }

        Self {
            observation_space,
            action_space,
            episodes: 0,
            batch_size: 32,
            learning_rate: 0.001,
            gamma: 0.99,
            eps_start: 0.9,
            eps_end: 0.05,
            eps_decay: 1000,
            replay_buffer: Vec::new(),
            steps_done: 0,
        }
    }

    pub fn choose_action(&mut self, state: &Vec<f32>) -> PlayerAction {
        let num_actions = self
            .action_space
            .n_discrete()
            .expect("Action space must be discrete");

        let current_eps = self.eps_end
            + (self.eps_start - self.eps_end)
                * (-1. * self.steps_done as f64 / self.eps_decay as f64)
                    .exp()
                    .max(0.0);

        let action_index = if rand::random::<f64>() < current_eps {
            // Explore: choose a random action from the action space.
            self.action_space.sample_index()
        } else {
            // Exploit: choose the best action based on the current Q-values for the state.
            // The Q-values would be calculated for each action_index from 0 to num_actions-1.
            (0..num_actions)
                .map(|idx| {
                    // Placeholder for Q-value calculation for action_idx.
                    // In a real implementation, this would involve a neural network forward pass:
                    // let q_value = self.model.predict(state, idx);
                    let q_value = rand::random::<f64>(); // Replace with actual Q-value
                    (idx, q_value)
                })
                .max_by(|(_, q1), (_, q2)| q1.partial_cmp(q2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or_else(|| {
                    eprintln!("Warning: Could not determine best action during exploitation, falling back to random sampling.");
                    self.action_space.sample_index()
                })
        };
        PlayerAction::from(action_index)
    }

    pub fn store_experience(
        &mut self,
        state: Vec<f64>,
        action: usize,
        reward: f64,
        next_state: Vec<f64>,
        done: bool,
    ) {
        self.replay_buffer
            .push((state, action, reward, next_state, done));
        if self.replay_buffer.len() > 10000 {
            self.replay_buffer.remove(0);
        }
    }

    pub fn train(&mut self) {
        // Placeholder for training logic
        if !self.replay_buffer.is_empty() {
            let (state, action, reward, next_state, done) = &self.replay_buffer[0];
            println!(
                "Training with state: {:?}, action: {}, reward: {}, next_state: {:?}, done: {}",
                state, action, reward, next_state, done
            );
        }
    }
}
