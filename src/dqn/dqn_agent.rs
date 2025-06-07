use crate::core::agent::{ActionSpace, PlayerAction};
use bevy::prelude::*;

#[derive(Resource, Debug)]
pub struct DQNAgent {
    observation_space: usize,
    action_space: ActionSpace,
    episodes: usize,
    batch_size: usize,
    learning_rate: f64,
    discount_factor: f64,
    exploration_rate: f64,
    exploration_decay: f64,
    replay_buffer: Vec<(Vec<f64>, usize, f64, Vec<f64>, bool)>,
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
            discount_factor: 0.99,
            exploration_rate: 1.0,
            exploration_decay: 0.995,
            replay_buffer: Vec::new(),
        }
    }

    pub fn choose_action(&mut self, state: &Vec<f32>) -> PlayerAction {
        let num_actions = self
            .action_space
            .n_discrete()
            .expect("Action space must be discrete");

        let action_index = if rand::random::<f64>() < self.exploration_rate {
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

    pub fn update_exploration_rate(&mut self) {
        if self.exploration_rate > 0.01 {
            self.exploration_rate *= self.exploration_decay;
        }
    }
}
