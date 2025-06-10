pub mod dqn_agent;
pub mod dqn_memory;
pub mod dqn_net;
pub mod dqn_optim;
pub mod dqn_training;

use crate::dqn::dqn_net::DQNModel;
use bevy::prelude::*;
use burn::optim::AdamConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use std::sync::{Arc, Mutex};

#[derive(Resource)]
pub struct PolicyNet<B: Backend>(pub Arc<Mutex<DQNModel<B>>>);

#[derive(Resource)]
pub struct TargetNet<B: Backend>(pub Arc<Mutex<DQNModel<B>>>);

#[derive(Resource)]
pub struct DqnOptimizer(pub AdamConfig);

#[derive(Resource, Default)]
pub struct EpisodeDone(pub bool);

#[derive(Resource, Clone)]
pub struct BurnDevice<B: Backend>(pub B::Device);

#[derive(Resource)]
pub struct ModelOptimizer<B: AutodiffBackend>(
    pub Arc<Mutex<OptimizerAdaptor<burn::optim::Adam, DQNModel<B>, B>>>,
);
