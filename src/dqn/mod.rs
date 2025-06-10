pub mod dqn_agent;
pub mod dqn_memory;
pub mod dqn_net;
pub mod dqn_optim;
pub mod dqn_training;

use crate::dqn::dqn_net::DQNModel;
use bevy::prelude::*;
use burn::optim::Adam;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

#[derive(Resource)]
pub struct PolicyNet<B: AutodiffBackend>(pub DQNModel<B>);

#[derive(Resource)]
pub struct TargetNet<B: AutodiffBackend>(pub DQNModel<B>);

#[derive(Resource)]
pub struct ModelOptimizer(pub Adam);

#[derive(Resource, Default)]
pub struct EpisodeDone(pub bool);

#[derive(Resource, Clone)]
pub struct BurnDevice<B: Backend>(pub B::Device);
