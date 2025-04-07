use crate::tensor::Tensor;
use super::parameter::{Parameter2D};
use super::Layer;


pub struct Linear {
    pub params: Parameter2D,
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

    fn forward(&self, inputs: &Tensor) -> Tensor {
        self.params.forward(inputs)
    }
}