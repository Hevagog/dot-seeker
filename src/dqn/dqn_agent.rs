use crate::core::agent::{ActionSpace, PlayerAction};
use bevy::prelude::*;
use burn::{prelude::*, tensor::cast::ToElement};

use crate::dqn::dqn_net::DQNModel;

#[derive(Resource, Debug)]
pub struct DQNAgent {
    pub observation_space: usize,
    pub action_space: ActionSpace,
    pub episodes: usize,
    pub steps_done: usize, // Counter for total steps taken (for epsilon decay)
    // Hyperparameters
    pub batch_size: usize,
    pub learning_rate: f64,
    pub gamma: f64,
    pub tau: f32,
    pub eps_start: f64,
    pub eps_end: f64,
    pub eps_decay: usize,
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
            steps_done: 0,
            batch_size: 32,
            learning_rate: 0.001,
            gamma: 0.99,
            tau: 0.005,
            eps_start: 0.9,
            eps_end: 0.05,
            eps_decay: 1000,
        }
    }

    fn current_epsilon(&self) -> f64 {
        self.eps_end
            + (self.eps_start - self.eps_end)
                * (-1. * self.steps_done as f64 / self.eps_decay as f64)
                    .exp()
                    .max(0.0)
    }

    pub fn choose_action<B: Backend>(
        &self,
        state_vec: &[f32],
        policy_net: &DQNModel<B>,
        device: &B::Device,
    ) -> PlayerAction {
        let action_index = if rand::random::<f64>() < self.current_epsilon() {
            // Explore: choose a random action
            self.action_space.sample_index()
        } else {
            // Exploit: use policy_net for inference
            let state_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(state_vec.to_vec(), [1, self.observation_space]),
                device,
            );
            let q_values = policy_net.forward(state_tensor);
            let action_tensor = q_values.argmax(1).squeeze::<1>(1);
            action_tensor.into_scalar().to_usize()
        };
        PlayerAction::from(action_index)
    }

    pub fn increment_step(&mut self) {
        self.steps_done += 1;
    }
}
