use crate::dqn::dqn_agent::*;
use crate::dqn::dqn_memory::*;
use crate::dqn::dqn_net::*;
use burn::prelude::*;
use burn::tensor::TensorData;

pub fn optimize_model<B: Backend>(
    dqn_memory: &mut DQNMemory,
    policy_net: &DQNModel<B>,
    target_net: &DQNModel<B>,
    dqn_agent: &mut DQNAgent,
    // optimizer: &mut burn::optim::Optimizer<B>,
) {
    if dqn_memory.len() < dqn_agent.batch_size {
        return; // Not enough samples to train
    }
    let device = &policy_net.linear1.weight.device();
    let transitions = dqn_memory.sample(dqn_agent.batch_size);
    let mut batch = Vec::new();
    for transition in transitions {
        batch.push((
            transition.state.clone(),
            transition.action,
            transition.reward,
            transition.next_state.clone(),
            transition.done,
        ));
    }
    let non_terminal_mask: Vec<bool> = batch.iter().map(|(_, _, _, _, done)| !done).collect();
    let non_terminal_next_states: Vec<Vec<f32>> = batch
        .iter()
        .filter(|(_, _, _, _, done)| !done)
        .map(|(_, _, _, next_state, _)| next_state.clone())
        .collect();

    let state_batch_vecs: Vec<Vec<f32>> = batch
        .iter()
        .map(|(state, _, _, _, _)| state.clone())
        .collect();
    let action_batch: Vec<usize> = batch.iter().map(|(_, action, _, _, _)| *action).collect();
    let reward_batch: Vec<f32> = batch.iter().map(|(_, _, reward, _, _)| *reward).collect();
    // let action_tensor = Tensor::<B, 1, Int>::from_floats(action_batch, device);
    // let reward_tensor = Tensor::<B, 1>::from_floats(reward_batch, device);
    let flattened_states: Vec<f32> = state_batch_vecs.into_iter().flatten().collect();

    let observation_space_size = policy_net.linear1.weight.dims()[1]; // Get input_shape from the model's first layer

    let state_data = TensorData::new(
        flattened_states,
        [dqn_agent.batch_size, observation_space_size],
    );
    let state_tensor = Tensor::<B, 2>::from_data(state_data, device);
    // Forward pass through the DQN model
    let q_values = policy_net.forward(state_tensor);

    // let state_action_values = select_actions(q_values, action_tensor.clone());

    // let next_q_values = target_net.forward(next_state_tensor.clone());
    // let next_state_max = next_q_values.max_dim(1);
    // let done_mask = Tensor::<B, 1, Bool>::from_floats(done_mask_as_u8, device);
    // let expected_state_action_values =
    //     reward_tensor.clone() + (next_state_max * dqn_agent.gamma) * done_mask.not();

    // let loss = MSELoss::default().forward(
    //     state_action_values.unsqueeze(),
    //     expected_state_action_values.unsqueeze(),
    // );

    // optimizer.backward_step(&loss);
}
