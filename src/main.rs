use bevy::prelude::*;
use bevy_rapier2d::prelude::*;
use rand;

mod core;
use core::agent::*;
use core::environment::Environment;
use core::environment::spawners::*;
use core::environment::systems::environment_system::*;

fn setup_environment_resources(mut commands: Commands) {
    let player_entity = spawn_player(&mut commands, Vec2::new(0.0, 0.0), Collider::ball(25.0));
    let goal_entity = spawn_goal(&mut commands, Vec2::new(200.0, 200.0));

    let wall_entities = vec![
        spawn_wall(
            &mut commands,
            Vec2::new(0.0, 300.0),
            Collider::cuboid(400.0, 20.0),
        ), // Top
        spawn_wall(
            &mut commands,
            Vec2::new(0.0, -300.0),
            Collider::cuboid(400.0, 20.0),
        ), // Bottom
        spawn_wall(
            &mut commands,
            Vec2::new(400.0, 0.0),
            Collider::cuboid(20.0, 300.0),
        ), // Right
        spawn_wall(
            &mut commands,
            Vec2::new(-400.0, 0.0),
            Collider::cuboid(20.0, 300.0),
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

// Dummy system to simulate agent choosing an action (replace with actual agent logic)
fn agent_choose_action_system(mut current_action: ResMut<Action>) {
    // For testing, cycle through actions or pick randomly
    // In a real scenario, your RL agent logic would set this.
    // This is just a placeholder.
    if rand::random::<f32>() < 0.05 {
        // Occasionally change action
        match rand::random::<u8>() % 4 {
            0 => current_action.0 = PlayerAction::Up,
            1 => current_action.0 = PlayerAction::Down,
            2 => current_action.0 = PlayerAction::Left,
            3 => current_action.0 = PlayerAction::Right,
            _ => (), // Should never happen
        }
    }
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
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::pixels_per_meter(100.0))
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_event::<ResetEvent>() // Register the reset event
        .add_systems(Startup, (setup_graphics, setup_environment_resources))
        // RL Loop Systems - Order can be important.
        // Using .chain() or .before/.after for explicit ordering.
        .add_systems(Update, agent_choose_action_system) // 1. Agent (or dummy) decides action
        .add_systems(Update, perform_action.after(agent_choose_action_system)) // 2. Apply action
        // 3. Physics step (Rapier runs automatically within Bevy's schedule)
        .add_systems(Update, state_extraction_system.after(perform_action)) // 4. Observe new state
        .add_systems(Update, reward_system.after(state_extraction_system)) // 5. Calculate reward
        // .add_systems(Update, print_ball_altitude)
        .add_systems(Update, check_and_trigger_reset_system.after(reward_system)) // 6. Check for reset conditions
        .add_systems(
            Update,
            reset_environment_system.run_if(on_event::<ResetEvent>),
        ) // 7. Reset if event occurs
        .run();
}

fn setup_graphics(mut commands: Commands) {
    commands.spawn(Camera2d);
}

fn print_ball_altitude(
    environment: Option<Res<Environment>>, // Make it optional as it might not exist on first frames
    positions: Query<&Transform, With<RigidBody>>,
) {
    if let Some(env) = environment {
        if let Ok(player_transform) = positions.get(env.player) {
            // println!("Player altitude: {}", player_transform.translation.y);
        }
    }
}
