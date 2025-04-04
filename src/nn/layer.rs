use super::parameter::{Parameter2D};
use super::Layer;

use crate::tensor::Tensor;

pub struct Linear {
    params: Parameter2D,
}

impl Layer for Linear {
    fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            params: Parameter2D::new(input_size, output_size),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.params.parameters()
    }

    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        self.params.forward(inputs)
    }
}