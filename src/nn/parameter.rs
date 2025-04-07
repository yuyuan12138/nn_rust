use rand::Rng;
use crate::tensor::{Tensor};
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
        let bias = Tensor::scalar(rand::random::<f64>() * 0.1);

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
        let mut rng = rand::rng();
        let xavier_std = (2.0 / (input_size as f64 + output_size as f64)).sqrt();

        let weights = Tensor::matrix(
            (0..output_size).map(|_| {
                (0..input_size).map(|_| {
                    let val: f64 = rng.random(); // [0,1) 随机数
                    (val - 0.5) * 2.0 * xavier_std // 转换到 [-xavier_std, xavier_std]
                }).collect()
            }).collect()
        );
        let bias = Tensor::vector(
            (0..output_size).map(|_| {
                let val: f64 = rng.random();
                (val - 0.5) * 0.01 // [-0.005, 0.005]
            }).collect()
        );

        Parameter2D{
            weights,
            bias,
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn forward(&self, inputs: &Tensor) -> Tensor {
        self.weights.matmul(inputs).add(&self.bias)
    }
}