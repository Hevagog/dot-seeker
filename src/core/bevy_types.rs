use crate::core::agent::PlayerAction;
use bevy::prelude::*;

#[derive(Resource)]
pub struct Environment {
    pub player: Entity,
    pub player_initial_spawn_position: Vec2,
    pub goal: Entity,
    pub walls: Vec<Entity>,
}

#[derive(Event, Debug)]
pub struct EpisodeDoneEvent;

#[derive(Resource, Default, Debug)]
pub struct EpisodeDoneFlag(pub bool);

#[derive(Resource, Default, Debug)]
pub struct RLState(pub Vec<f32>);

#[derive(Resource, Default, Debug)]
pub struct CurrentReward(pub f32);

#[derive(Resource, Default, Debug)]
pub struct Action(pub PlayerAction);
