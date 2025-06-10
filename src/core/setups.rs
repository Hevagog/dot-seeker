use bevy::prelude::*;
use bevy::sprite::{ColorMaterial, MeshMaterial2d};

use bevy_rapier2d::prelude::*;

use crate::core::agent::*;
use crate::core::environment::spawners::*;
use crate::core::environment::{CurrentReward, Environment, RLState};
use crate::dqn::dqn_agent::DQNAgent;

pub fn setup_environment_resources(
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

pub fn setup_agent(mut commands: Commands) {
    let observation_space = 6; // Example observation space size
    let action_space = ActionSpace::Discrete(PlayerAction::COUNT); // Example action space size (Up, Down, Left, Right)
    let agent = DQNAgent::new(observation_space, action_space);
    commands.insert_resource(agent);
}
