# Dot Seeker - Reinforcement Learning Environment

Dot Seeker is a 2D reinforcement learning environment built with the [Bevy game engine](https://bevyengine.org/) and the [Rapier physics engine](https://rapier.rs/) in Rust.

The primary goal of this project is to create a simple yet flexible environment where an agent (the "player") learns to navigate a 2D space to reach a goal dot while potentially avoiding obstacles.

## Tech Stack

*   **Rust**: The programming language used for the entire project.
*   **Bevy Engine**: A refreshingly simple data-driven game engine built in Rust. It provides the Entity Component System (ECS) framework, rendering, input handling, and other game-related functionalities.
*   **Rapier2D**: A 2D physics engine for Rust, used here to handle collisions and movement dynamics for the player and walls.
* **Burn**: A flexible and comprehensive deep learning framework in Rust, used for implementing the DQN agent's neural networks and training process.

## DQN Implementation

The reinforcement learning agent uses a Deep Q-Network (DQN) algorithm, implemented using the Burn framework. The core components of this solution are:

*   **Agent ([`dqn::dqn_agent::DQNAgent`](src/dqn/dqn_agent.rs))**: Manages the agent's hyperparameters (like learning rate, discount factor gamma, epsilon for exploration-exploitation) and decides actions based on the current policy (epsilon-greedy). It's defined in [src/dqn/dqn_agent.rs](src/dqn/dqn_agent.rs).
*   **Neural Network ([`dqn::dqn_net::DQNModel`](src/dqn/dqn_net.rs))**: A multi-layer perceptron (MLP) defined in [src/dqn/dqn_net.rs](src/dqn/dqn_net.rs). It takes the environment state as input and outputs Q-values for each possible action. The configuration for this model is [`dqn::dqn_net::ModelConfig`](src/dqn/dqn_net.rs). Two instances of this model are used: a policy network for action selection and a target network for stable Q-value estimation during training.
*   **Replay Memory ([`dqn::dqn_memory::DQNMemory`](src/dqn/dqn_memory.rs))**: Stores experiences (state, action, reward, next_state, done) in a replay buffer. This allows the agent to learn from a diverse set of past experiences, breaking correlations in sequential data. Implemented in [src/dqn/dqn_memory.rs](src/dqn/dqn_memory.rs).
*   **Optimization ([`dqn::dqn_optim`](src/dqn/dqn_optim.rs))**: The [`dqn::dqn_optim::optimize_model`](src/dqn/dqn_optim.rs) function handles the learning step. It samples a batch of experiences from the replay memory, calculates the loss (Huber loss) between the predicted Q-values (from the policy network) and the target Q-values (derived from the target network), and updates the policy network's weights using an Adam optimizer. The target network is updated slowly towards the policy network using [`dqn::dqn_optim::polyak_update`](src/dqn/dqn_optim.rs).
*   **Training Loop ([`dqn::dqn_training::run_training_episodes`](src/dqn/dqn_training.rs))**: Orchestrates the interaction between the agent and the Bevy environment. For each episode, it resets the environment, then iteratively:
    1.  The agent chooses an action using the policy network.
    2.  The action is performed in the Bevy environment ([`core::environment::systems::environment_system::perform_action_system`](src/core/environment/systems/environment_system.rs)).
    3.  The environment transitions to a new state, and a reward is received ([`core::environment::systems::environment_system::observe_system`](src/core/environment/systems/environment_system.rs) and [`core::environment::systems::environment_system::reward_system`](src/core/environment/systems/environment_system.rs)).
    4.  The experience is stored in the replay memory.
    5.  The `optimize_model` function is called to update the policy network.
    6.  The target network is updated.
    This process is detailed in [src/dqn/dqn_training.rs](src/dqn/dqn_training.rs).
*   **Bevy Integration**: The DQN components are integrated into the Bevy ECS as resources (e.g., [`DQNAgent`](src/dqn/dqn_agent.rs), [`PolicyNet`](src/dqn/mod.rs), [`TargetNet`](src/dqn/mod.rs), [`DQNMemory`](src/dqn/dqn_memory.rs)). Bevy systems manage the environment's state transitions, reward calculation, and observation generation, as seen in [src/main.rs](src/main.rs) and [src/core/environment/systems/environment_system.rs](src/core/environment/systems/environment_system.rs).

## Running the Project Headless

To run the Dot Seeker project headless:

```bash
cargo run --no-default-features --features headless
```