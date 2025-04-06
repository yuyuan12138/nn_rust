use crate::tensor::{Tensor, TensorValue};
use super::Layer;

pub struct Parameter1D {
    pub weights: Tensor,
    pub bias: Tensor,
}

pub struct Parameter2D {
    pub weights: Tensor,
    pub bias: Tensor,
}

impl Layer for Parameter1D {
    fn new(input_size: usize, output_size: usize) -> Self {
        assert_eq!(input_size, output_size, "input_size and output_size must be the same in Parameter1D.");
        let weights = Tensor::vector(vec![rand::random::<f64>() * 0.1; input_size]);
        let bias = Tensor::vector(vec![rand::random::<f64>() * 0.1; input_size]);

        Parameter1D{
            weights,
            bias,
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.multiply(&self.weights).add(&self.bias)
    }
}

impl Layer for Parameter2D {
    fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Tensor::matrix(vec![vec![rand::random::<f64>() * 0.1; input_size]; output_size]);
        let bias = Tensor::vector(vec![rand::random::<f64>() * 0.1; output_size]);

        Parameter2D{
            weights,
            bias,
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        self.weights.matmul(inputs).add(&self.bias).sigmoid()
    }
}