use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct DQNModel<B: Backend> {
    pub linear1: Linear<B>,
    pub relu1: Relu,
    pub linear2: Linear<B>,
    pub relu2: Relu,
    pub linear3: Linear<B>,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    input_shape: usize,
    output_shape: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DQNModel<B> {
        DQNModel {
            linear1: LinearConfig::new(self.input_shape, 64).init(device),
            relu1: Relu::new(),
            linear2: LinearConfig::new(128, 64).init(device),
            relu2: Relu::new(),
            linear3: LinearConfig::new(64, self.output_shape).init(device),
        }
    }
}

impl<B: Backend> DQNModel<B> {
    /// # Shapes
    ///   - Images [batch_size, ObservationSpace]
    ///   - Output [batch_size, ActionSpace]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x: Tensor<B, 2> = self.linear1.forward(input);
        let x = self.relu1.forward(x);
        let x = self.linear2.forward(x);
        let x = self.relu2.forward(x);
        self.linear3.forward(x)
    }
}
