use bevy::prelude::*;
use rand::Rng;

pub struct Memory_Record {
    pub state: Vec<f32>,
    pub action: usize,
    pub next_state: Vec<f32>,
    pub reward: f32,
    pub done: bool,
}

#[derive(Resource)]
pub struct DQNMemory {
    pub replay_buffer: Vec<Memory_Record>,
    pub max_size: usize,
}

impl DQNMemory {
    pub fn new(max_size: usize) -> Self {
        DQNMemory {
            replay_buffer: Vec::new(),
            max_size,
        }
    }

    pub fn store_experience(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        let record = Memory_Record {
            state,
            action,
            next_state,
            reward,
            done,
        };
        self.replay_buffer.push(record);
        if self.replay_buffer.len() > self.max_size {
            self.replay_buffer.remove(0); // Maintain a maximum size
        }
    }

    pub fn sample(&self, batch_size: usize) -> Vec<&Memory_Record> {
        let mut rng = rand::rng();
        (0..batch_size)
            .map(|_| {
                let index = rng.random_range(0..self.replay_buffer.len());
                &self.replay_buffer[index]
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.replay_buffer.len()
    }
}
