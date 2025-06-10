use crate::core::agent::{Action, PlayerAction};
use crate::core::environment::components::*;
use crate::core::environment::{CurrentReward, Environment, RLState};

use bevy::prelude::*;
use bevy_rapier2d::prelude::*;

pub fn observe(
    environment: Res<Environment>,
    player_query: Query<(&Transform, &Velocity), With<Player>>, // Query player by its components
    goal_query: Query<&Transform, With<Goal>>,                  // Query goal by its components
    mut rl_state: ResMut<RLState>,
) {
    rl_state.0.clear();

    if let Ok((player_transform, player_velocity)) = player_query.get(environment.player) {
        rl_state.0.push(player_transform.translation.x);
        rl_state.0.push(player_transform.translation.y);
        rl_state.0.push(player_velocity.linvel.x);
        rl_state.0.push(player_velocity.linvel.y);
    } else {
        rl_state.0.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
    }

    if let Ok(goal_transform) = goal_query.get(environment.goal) {
        rl_state.0.push(goal_transform.translation.x);
        rl_state.0.push(goal_transform.translation.y);
    } else {
        rl_state.0.extend_from_slice(&[0.0, 0.0]);
    }
}

pub fn reward_system(
    environment: Res<Environment>,
    player_query: Query<&Transform, With<Player>>,
    goal_query: Query<&Transform, With<Goal>>,
    mut current_reward: ResMut<CurrentReward>,
) {
    if let (Ok(player_transform), Ok(goal_transform)) = (
        player_query.get(environment.player),
        goal_query.get(environment.goal),
    ) {
        let player_pos = player_transform.translation.truncate();
        let goal_pos = goal_transform.translation.truncate();
        let distance = player_pos.distance(goal_pos);
        if distance < 0.5 {
            current_reward.0 = 1.0;
        } else {
            current_reward.0 = -distance * 0.1;
        }
    }
}

pub fn reset_environment_system(
    environment: Res<Environment>,
    mut queries: ParamSet<(
        Query<(&mut Transform, &mut Velocity), With<Player>>,
        Query<&mut Transform, With<Goal>>,
    )>,
) {
    if let Ok((mut player_transform, mut player_velocity)) =
        queries.p0().get_mut(environment.player)
    {
        player_transform.translation = Vec3::new(
            environment.player_initial_spawn_position.x,
            environment.player_initial_spawn_position.y,
            0.0,
        );
        player_velocity.linvel = Vec2::ZERO;
        player_velocity.angvel = 0.0;
    }
    if let Ok(mut goal_transform) = queries.p1().get_mut(environment.goal) {
        // let new_x = (rand::random::<f32>() - 0.5) * 20.0;
        // let new_y = (rand::random::<f32>() - 0.5) * 20.0;
        // goal_transform.translation = Vec3::new(new_x, new_y, 0.0);
        goal_transform.translation = Vec3::new(200.0, 200.0, 0.0);
    }
}

pub fn perform_action(
    current_action: Res<Action>,
    mut player_query: Query<&mut Velocity, With<Player>>,
    environment: Res<Environment>,
) {
    let action = current_action.0;
    if let Ok(mut rapier_velocity) = player_query.get_mut(environment.player) {
        const MOVE_SPEED: f32 = 40.0;
        match action {
            PlayerAction::Up => rapier_velocity.linvel = Vec2::new(0.0, MOVE_SPEED),
            PlayerAction::Down => rapier_velocity.linvel = Vec2::new(0.0, -MOVE_SPEED),
            PlayerAction::Left => rapier_velocity.linvel = Vec2::new(-MOVE_SPEED, 0.0),
            PlayerAction::Right => rapier_velocity.linvel = Vec2::new(MOVE_SPEED, 0.0),
        }
    }
}
