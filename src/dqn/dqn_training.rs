use crate::dqn::dqn_agent::*;
use crate::dqn::dqn_memory::*;
use crate::dqn::dqn_optim::*;
use crate::dqn::*;

use crate::core::bevy_types::{
    Action as BevyAction, CurrentReward, EpisodeDoneEvent, EpisodeDoneFlag, RLState,
};
use bevy::ecs::system::SystemParam;
use bevy::ecs::system::SystemState;
use bevy::prelude::*;
use bevy::prelude::{App, Res, ResMut};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use std::marker::PhantomData;

#[derive(SystemParam)]
struct OptimizeModelParams<'w, 's, B: AutodiffBackend + 'static> {
    memory: ResMut<'w, DQNMemory>,
    agent_config: Res<'w, DQNAgent>,
    policy_net: ResMut<'w, PolicyNet<B>>,
    target_net: Res<'w, TargetNet<B>>,
    optimizer: ResMut<'w, ModelOptimizer<B>>,
    _phantom_s: PhantomData<&'s ()>,
    _phantom_b: PhantomData<B>,
}

pub fn run_training_episodes<B: AutodiffBackend>(app: &mut App, num_episodes: usize)
where
    B: Backend,
{
    let mut optimize_params_state: SystemState<OptimizeModelParams<B>> =
        SystemState::new(app.world_mut());

    for _episode_i in 0..num_episodes {
        // --- RESET ENVIRONMENT ---
        app.world_mut().send_event(EpisodeDoneEvent);
        // Reset our loop's done flag, which is read from EpisodeDoneFlag resource
        app.world_mut()
            .get_resource_mut::<EpisodeDoneFlag>()
            .unwrap()
            .0 = false;
        app.update(); // Run Bevy systems: reset_environment_system, then observe_system

        let mut current_state_vec = app.world().get_resource::<RLState>().unwrap().0.clone();
        let mut total_episode_reward = 0.0;

        loop {
            // --- AGENT CHOOSES ACTION ---
            let chosen_action = {
                let agent_res = app.world().get_resource::<DQNAgent>().unwrap();
                let policy_net_res = app.world().get_resource::<PolicyNet<B>>().unwrap();
                let device_res = app.world().get_resource::<BurnDevice<B>>().unwrap();

                let policy_guard = policy_net_res.0.lock().unwrap(); // Lock Mutex
                agent_res.choose_action(&current_state_vec, &*policy_guard, &device_res.0)
            };

            // --- PERFORM ACTION IN BEVY ENVIRONMENT ---
            app.world_mut().get_resource_mut::<BevyAction>().unwrap().0 = chosen_action;
            app.update(); // Run Bevy systems: perform_action, physics, observe, reward, check_done

            // --- OBSERVE RESULTS FROM BEVY ---
            let next_state_vec = app.world().get_resource::<RLState>().unwrap().0.clone();
            let reward = app.world().get_resource::<CurrentReward>().unwrap().0;
            let done = app.world().get_resource::<EpisodeDoneFlag>().unwrap().0;

            total_episode_reward += reward;

            // --- STORE EXPERIENCE ---
            {
                let mut memory_res = app.world_mut().get_resource_mut::<DQNMemory>().unwrap();
                memory_res.store_experience(
                    current_state_vec.clone(),
                    chosen_action.into(),
                    reward,
                    next_state_vec.clone(),
                    done,
                );
            }

            current_state_vec = next_state_vec;

            // --- AGENT LEARNING STEP (OPTIMIZE MODEL) ---
            {
                let mut params = optimize_params_state.get_mut(app.world_mut());

                if params.memory.len() >= params.agent_config.batch_size {
                    let mut policy_guard_mut = params.policy_net.0.lock().unwrap();
                    let target_guard_immut = params.target_net.0.lock().unwrap();
                    let mut optimizer_guard_mut = params.optimizer.0.lock().unwrap();

                    optimize_model(
                        &mut params.memory,
                        &mut *policy_guard_mut,
                        &*target_guard_immut,
                        &params.agent_config,
                        &mut *optimizer_guard_mut,
                    );
                }
            }

            app.world_mut()
                .get_resource_mut::<DQNAgent>()
                .unwrap()
                .increment_step();

            if done {
                break;
            }
        }

        {
            // TODO: switch to system state
            let agent_config = app.world().get_resource::<DQNAgent>().unwrap();
            let policy_net_res_immut = app.world().get_resource::<PolicyNet<B>>().unwrap();
            let target_net_res_mut = app.world().get_resource::<TargetNet<B>>().unwrap();

            let policy_guard_immut = policy_net_res_immut.0.lock().unwrap();
            let mut target_guard_mut = target_net_res_mut.0.lock().unwrap();

            polyak_update(
                &*policy_guard_immut,
                &mut *target_guard_mut,
                agent_config.tau,
            );
        }

        let mut agent_mut_res = app.world_mut().get_resource_mut::<DQNAgent>().unwrap();
        agent_mut_res.episodes += 1;
        println!(
            "Episode: {}, Total Reward: {:.2}, Steps Done: {}",
            agent_mut_res.episodes, total_episode_reward, agent_mut_res.steps_done
        );
    }
}
