#![recursion_limit = "256"]

use crate::dqn::dqn_net::DQNModel;
use bevy::prelude::*;

use bevy_rapier2d::prelude::*;
use burn::optim::AdamConfig;
use burn::prelude::*;

mod core;
mod dqn;

use crate::core::agent::{ActionSpace, PlayerAction};
use crate::core::setups::{setup_agent, setup_environment_resources};
use crate::dqn::dqn_agent::DQNAgent;
use crate::dqn::dqn_memory::DQNMemory;
use crate::dqn::dqn_net::ModelConfig;
use crate::dqn::dqn_optim::polyak_update;
use crate::dqn::{PolicyNet, TargetNet};
use std::sync::{Arc, Mutex};

use crate::dqn::*;
use burn::backend::{Autodiff, Wgpu};
use core::bevy_types::{Action, CurrentReward, EpisodeDoneEvent, EpisodeDoneFlag, RLState};
use core::environment::systems::environment_system::*;

type MyBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

fn check_done_system(
    reward: Res<CurrentReward>,
    mut ev_reset: EventWriter<EpisodeDoneEvent>,
    mut episode_done_flag: ResMut<EpisodeDoneFlag>, // Add this parameter
) {
    if reward.0 >= 1.0 {
        println!("Goal reached! Requesting reset and setting flag.");
        episode_done_flag.0 = true;
        ev_reset.write(EpisodeDoneEvent);
    }
}

fn main() {
    // Bevy app setup
    let mut app = App::new();
    // Initialize the learning agent and environment and backend

    let burn_device = <MyAutodiffBackend as Backend>::Device::default(); // Use MyAutodiffBackend for device if model init needs it

    let observation_space = 6;
    let action_space = ActionSpace::Discrete(PlayerAction::COUNT); // Up, Down, Left, Right
    let agent = DQNAgent::new(observation_space, action_space);

    app.insert_resource(agent)
        .insert_resource(DQNMemory::new(10000));

    let model_config = ModelConfig {
        input_shape: observation_space,
        output_shape: PlayerAction::COUNT,
    };

    let policy_model = model_config.init::<MyAutodiffBackend>(&burn_device);
    let target_model = model_config.init::<MyAutodiffBackend>(&burn_device);

    // It's good practice to ensure the target network starts as a copy of the policy network
    let mut target_model_for_polyak = target_model.clone();
    polyak_update(&policy_model, &mut target_model_for_polyak, 1.0);

    app.insert_resource(PolicyNet(Arc::new(Mutex::new(policy_model))))
        .insert_resource(TargetNet(Arc::new(Mutex::new(target_model_for_polyak))));

    let adam_config = AdamConfig::new();
    app.insert_resource(ModelOptimizer(Arc::new(Mutex::new(
        adam_config.init::<MyAutodiffBackend, DQNModel<MyAutodiffBackend>>(),
    ))))
    .insert_resource(RLState(vec![0.0; observation_space]))
    .insert_resource(CurrentReward(0.0))
    .insert_resource(Action(PlayerAction::default()))
    .insert_resource(BurnDevice::<MyAutodiffBackend>(burn_device))
    .insert_resource(EpisodeDoneFlag(false));

    #[cfg(feature = "gui")]
    {
        app.add_plugins(DefaultPlugins)
            .add_plugins(RapierDebugRenderPlugin::default());
    }

    #[cfg(feature = "headless")]
    {
        app.add_plugins(MinimalPlugins);
    }

    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_event::<EpisodeDoneEvent>()
        .add_systems(
            Startup,
            (setup_graphics, setup_environment_resources, setup_agent),
        )
        .add_systems(
            Update,
            (
                perform_action_system,
                observe_system.after(perform_action_system),
                reward_system.after(observe_system),
                check_done_system.after(reward_system),
                display_debug_info.after(check_done_system),
            )
                .chain(),
        )
        .add_systems(
            Update,
            reset_environment_system.run_if(on_event::<EpisodeDoneEvent>),
        );

    // ------ Start Training ------
    let num_training_episodes = 1000;
    crate::dqn::dqn_training::run_training_episodes::<MyAutodiffBackend>(
        &mut app,
        num_training_episodes,
    );
}

fn setup_graphics(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn display_reward(reward: Res<CurrentReward>) {
    println!("Current Reward: {}", reward.0);
}

fn display_observation(state: Res<RLState>) {
    println!("Current State: {:?}", state.0);
}

fn display_debug_info(reward: Res<CurrentReward>, state: Res<RLState>) {
    display_reward(reward);
    display_observation(state);
}

pub fn dqn_agent_decision_system(
    agent: Res<DQNAgent>,
    current_rl_state: Res<RLState>,
    policy_net_resource: Res<PolicyNet<MyAutodiffBackend>>,
    burn_device_resource: Res<BurnDevice<MyAutodiffBackend>>,
    mut current_action: ResMut<Action>,
) {
    let policy_net_guard = policy_net_resource.0.lock().unwrap();

    let policy_model = &*policy_net_guard;
    let device = &burn_device_resource.0;

    // Choose action
    let chosen_player_action = agent.choose_action(&current_rl_state.0, policy_model, device);

    current_action.0 = chosen_player_action;
}
