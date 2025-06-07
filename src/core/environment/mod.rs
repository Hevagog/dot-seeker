pub mod components;
pub mod spawners;
pub mod systems;

use bevy::prelude::*;

#[derive(Resource)]
pub struct Environment {
    pub player: Entity,
    pub player_initial_spawn_position: Vec2,
    pub goal: Entity,
    pub walls: Vec<Entity>,
}
