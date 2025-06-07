# Dot Seeker - Reinforcement Learning Environment

Dot Seeker is a 2D reinforcement learning environment built with the [Bevy game engine](https://bevyengine.org/) and the [Rapier physics engine](https://rapier.rs/) in Rust.

The primary goal of this project is to create a simple yet flexible environment where an agent (the "player") learns to navigate a 2D space to reach a goal dot while potentially avoiding obstacles.

## Tech Stack

*   **Rust**: The programming language used for the entire project.
*   **Bevy Engine**: A refreshingly simple data-driven game engine built in Rust. It provides the Entity Component System (ECS) framework, rendering, input handling, and other game-related functionalities.
*   **Rapier2D**: A 2D physics engine for Rust, used here to handle collisions and movement dynamics for the player and walls.
* **Burn**: TODO


## Running the Project Headless

To run the Dot Seeker project headless:

```bash
cargo run --no-default-features --features headless
```