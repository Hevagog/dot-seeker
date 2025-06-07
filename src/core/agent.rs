use bevy::prelude::Resource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlayerAction {
    #[default]
    Up,
    Down,
    Left,
    Right,
}

#[derive(Resource, Default)]
pub struct Action(pub PlayerAction);
