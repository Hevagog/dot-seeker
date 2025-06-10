use crate::dqn::dqn_agent::*;
use crate::dqn::dqn_memory::*;
use crate::dqn::dqn_net::*;
use burn::module::Module;
use burn::nn::loss::{HuberLossConfig, Reduction};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::TensorData;
use burn::tensor::backend::AutodiffBackend;

use burn::tensor::{Int, Tensor};

pub fn optimize_model<B: AutodiffBackend>(
    dqn_memory: &mut DQNMemory,
    policy_net: &mut DQNModel<B>,
    target_net: &DQNModel<B>,
    dqn_agent: &DQNAgent,
    optimizer: &mut OptimizerAdaptor<burn::optim::Adam, DQNModel<B>, B>,
) {
    if dqn_memory.len() < dqn_agent.batch_size {
        return;
    }

    let device = &policy_net.linear1.weight.device();
    let transitions = dqn_memory.sample(dqn_agent.batch_size);

    // Build batches
    let mut states = Vec::new();
    let mut actions = Vec::new();
    let mut rewards = Vec::new();
    let mut next_states = Vec::new();
    let mut mask_indices = Vec::new();

    for (i, tr) in transitions.into_iter().enumerate() {
        states.extend(tr.state.clone());
        actions.push(tr.action as i64);
        rewards.push(tr.reward);
        if !tr.done {
            mask_indices.push(i as i64);
            next_states.extend(tr.next_state.clone());
        }
    }

    let b = dqn_agent.batch_size;
    let obs_size = policy_net.linear1.weight.dims()[1] as usize;

    // Tensors
    let state = Tensor::<B, 2>::from_data(TensorData::new(states, [b, obs_size]), device);
    let action = Tensor::<B, 2, Int>::from_data(TensorData::new(actions, [b, 1]), device);
    let reward = Tensor::<B, 2>::from_data(TensorData::new(rewards, [b, 1]), device);

    let mut next_state = Tensor::<B, 2>::zeros([b, obs_size], device);
    if !mask_indices.is_empty() {
        let idx = Tensor::<B, 2, Int>::from_data(
            TensorData::new(mask_indices.clone(), [mask_indices.len(), 1]),
            device,
        );
        let tmp = Tensor::<B, 2>::from_data(
            TensorData::new(next_states, [mask_indices.len(), obs_size]),
            device,
        );
        next_state = next_state.scatter(0, idx, tmp);
    }

    // Q(s,a)
    let q_values = policy_net.forward(state.clone());
    let state_action = q_values.gather(1, action);

    // Q_target
    let next_q = target_net.forward(next_state);
    let next_max = next_q.max_dim(1).unsqueeze_dim::<2>(1); // shape [b,1]
    let target = reward + next_max * dqn_agent.gamma;

    // Compute Huber loss
    let huber = HuberLossConfig::new(1.0).init();
    let loss = huber.forward(state_action, target, Reduction::Mean);

    // Backward pass
    let gradients_struct = loss.backward();
    // Create GradientsParams from the raw gradients and the policy network
    let grads_for_optimizer = GradientsParams::from_grads(gradients_struct, &*policy_net);

    // Perform optimization step
    // The optim_adaptor.step method takes ownership of the model and returns the updated model.
    // policy_net.clone() is used as step consumes the model.
    *policy_net = optimizer.step(
        dqn_agent.learning_rate,
        policy_net.clone(),
        grads_for_optimizer,
    );
}

pub fn polyak_update<B: AutodiffBackend>(source: &DQNModel<B>, target: &mut DQNModel<B>, tau: f32) {
    fn update_tensor<B: AutodiffBackend, const D: usize>(
        src: &Tensor<B, D>,
        tgt: &mut Tensor<B, D>,
        tau: f32,
    ) where
        Tensor<B, D>: Clone,
    {
        // new_tgt = src * tau + tgt * (1 - tau)
        let updated = src.clone().mul_scalar(tau) + tgt.clone().mul_scalar(1.0 - tau);
        *tgt = updated;
    }

    // Manually update each layer's weight and bias
    update_tensor(
        &source.linear1.weight.val(),
        &mut target.linear1.weight.val(),
        tau,
    );
    if let (Some(src_b_param), Some(tgt_b_param)) = (&source.linear1.bias, &mut target.linear1.bias)
    {
        update_tensor(&src_b_param.val(), &mut tgt_b_param.val(), tau);
    }

    // Linear 2
    update_tensor(
        &source.linear2.weight.val(),
        &mut target.linear2.weight.val(),
        tau,
    );
    if let (Some(src_b_param), Some(tgt_b_param)) = (&source.linear2.bias, &mut target.linear2.bias)
    {
        update_tensor(&src_b_param.val(), &mut tgt_b_param.val(), tau);
    }

    // Linear 3
    update_tensor(
        &source.linear3.weight.val(),
        &mut target.linear3.weight.val(),
        tau,
    );
    if let (Some(src_b_param), Some(tgt_b_param)) = (&source.linear3.bias, &mut target.linear3.bias)
    {
        update_tensor(&src_b_param.val(), &mut tgt_b_param.val(), tau);
    }
}
