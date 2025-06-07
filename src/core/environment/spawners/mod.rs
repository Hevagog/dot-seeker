use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

use crate::core::environment::components;

pub fn spawn_player(
    commands: &mut Commands,
    initial_spawn_position: Vec2,
    collider: Collider,
) -> Entity {
    commands
        .spawn((
            components::PlayerBundle::new(collider),
            Transform::from_xyz(initial_spawn_position.x, initial_spawn_position.y, 0.0),
            Velocity::zero(),
        ))
        .id()
}

pub fn spawn_wall(commands: &mut Commands, position: Vec2, collider: Collider) -> Entity {
    commands
        .spawn(components::WallBundle::new(position, collider))
        .id()
}

pub fn spawn_goal(commands: &mut Commands, position: Vec2) -> Entity {
    commands.spawn(components::GoalBundle::new(position)).id()
}
