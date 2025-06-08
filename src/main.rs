use bevy::prelude::*;
use bevy::sprite::{ColorMaterial, MeshMaterial2d};

use bevy_rapier2d::prelude::*;

mod core;
mod dqn;
use core::agent::*;
use core::environment::Environment;
use core::environment::spawners::*;
use core::environment::systems::environment_system::*;
use dqn::dqn_agent::DQNAgent;

fn setup_environment_resources(
    mut commands: Commands,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<ColorMaterial>>>,
) {
    let player_entity = spawn_player(&mut commands, Vec2::new(0.0, 0.0), Collider::ball(10.0));
    let goal_entity = spawn_goal(&mut commands, Vec2::new(200.0, 200.0));

    if let (Some(mut meshes_res), Some(mut materials_res)) = (meshes, materials) {
        commands.entity(goal_entity).insert((
            Mesh2d(meshes_res.add(Circle::new(10.0))),
            MeshMaterial2d(materials_res.add(Color::hsl(120.0, 1.0, 0.5))),
        ));
    }

    let wall_entities = vec![
        spawn_wall(
            &mut commands,
            Vec2::new(0.0, 300.0),
            Collider::cuboid(400.0, 10.0),
        ), // Top
        spawn_wall(
            &mut commands,
            Vec2::new(0.0, -300.0),
            Collider::cuboid(400.0, 10.0),
        ), // Bottom
        spawn_wall(
            &mut commands,
            Vec2::new(400.0, 0.0),
            Collider::cuboid(10.0, 300.0),
        ), // Right
        spawn_wall(
            &mut commands,
            Vec2::new(-400.0, 0.0),
            Collider::cuboid(10.0, 300.0),
        ), // Left
    ];

    commands.insert_resource(Environment {
        player: player_entity,
        player_initial_spawn_position: Vec2::new(0.0, 0.0),
        walls: wall_entities,
        goal: goal_entity,
    });

    commands.insert_resource(Action::default());
    commands.insert_resource(RLState::default());
    commands.insert_resource(CurrentReward::default());
}

fn setup_agent(mut commands: Commands) {
    let observation_space = 6; // Example observation space size
    let action_space = ActionSpace::Discrete(PlayerAction::COUNT); // Example action space size (Up, Down, Left, Right)
    let agent = DQNAgent::new(observation_space, action_space);
    commands.insert_resource(agent);
}

fn check_and_trigger_reset_system(
    reward: Res<CurrentReward>,
    mut ev_reset: EventWriter<ResetEvent>,
) {
    if reward.0 >= 1.0 {
        // Assuming reward 1.0 means goal reached
        println!("Goal reached! Requesting reset.");
        ev_reset.write(ResetEvent);
    }
}

#[derive(Event)]
struct ResetEvent;

fn main() {
    let mut app = App::new();

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
        .add_event::<ResetEvent>()
        .add_systems(
            Startup,
            (setup_graphics, setup_environment_resources, setup_agent),
        )
        .add_systems(
            Update,
            (
                // 1. Observe current state (ensure 'observe' runs before agent decision)
                observe,
                // 2. Agent uses observed state to decide action
                //    Using .after() to ensure 'observe' has updated RLState
                dqn_agent_decision_system.after(observe),
                // 3. Apply the chosen action to the environment
                perform_action.after(dqn_agent_decision_system),
                // 4. Calculate reward based on the action's outcome
                reward_system.after(perform_action),
                // 5. Agent learns from the experience (state, action, reward, next_state)
                //    This system would need access to:
                //    - DQNAgent
                //    - The state before the action (or store it)
                //    - The action taken (from Action resource)
                //    - The reward received (from CurrentReward resource)
                //    - The new state after the action (from RLState, after a new observe, or pass explicitly)
                //    - Done flag
                //    For simplicity, let's assume a basic train call for now.
                //    A more complete agent_learn_system would be more complex.
                // agent_learn_system.after(reward_system), // Placeholder for agent training
                // 6. Display debug info
                display_debug_info.after(reward_system),
                // 7. Check for reset conditions
                check_and_trigger_reset_system.after(reward_system),
            )
                .chain(),
        )
        .add_systems(
            Update,
            reset_environment_system.run_if(on_event::<ResetEvent>),
        )
        .run();
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
    mut agent: ResMut<DQNAgent>,
    current_state: Res<RLState>,
    mut current_action: ResMut<Action>,
) {
    let action = agent.choose_action(&current_state.0);
    current_action.0 = action;
}
