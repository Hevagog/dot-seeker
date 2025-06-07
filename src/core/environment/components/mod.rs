use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Goal;

#[derive(Component)]
pub struct Wall;

#[derive(Bundle)]
pub struct PlayerBundle {
    pub player: Player,
    pub rigid_body: RigidBody,
    pub collider: Collider,
}

impl PlayerBundle {
    pub fn new(collider: Collider) -> Self {
        Self {
            player: Player,
            rigid_body: RigidBody::Dynamic,
            collider: collider, //Collider::ball(50.0)
        }
    }
}

#[derive(Bundle)]
pub struct WallBundle {
    pub wall: Wall,
    pub transform: Transform,
    pub rigid_body: RigidBody,
    pub collider: Collider,
}

impl WallBundle {
    pub fn new(position: Vec2, collider: Collider) -> Self {
        Self {
            wall: Wall,
            transform: Transform::from_xyz(position.x, position.y, 0.0),
            rigid_body: RigidBody::Fixed,
            collider: collider, // Collider::cuboid(size.x / 2.0, size.y / 2.0),
        }
    }
}

#[derive(Bundle)]
pub struct GoalBundle {
    pub goal: Goal,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
}

impl GoalBundle {
    pub fn new(position: Vec2) -> Self {
        Self {
            goal: Goal,
            transform: Transform::from_xyz(position.x, position.y, 0.0),
            global_transform: GlobalTransform::default(),
        }
    }
}
