use crate::tensor::Tensor;
use anyhow::Result;
pub mod layer;
pub mod parameter;
pub mod optimizer;

pub trait Layer {
    fn new(input_size: usize, output_size: usize) -> Self;
    fn parameters(&self) -> Vec<&Tensor>;
    fn forward(&self, inputs: &Tensor) -> Result<Tensor>;

}

pub trait Optimizer {
    fn step(&self, params: &[&Tensor]);

    fn zero_grad(&self, params: &[&Tensor]);
}