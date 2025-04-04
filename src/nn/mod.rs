use crate::tensor::Tensor;
pub mod layer;
pub mod parameter;
pub mod operation;

pub trait Layer {
    fn new(input_size: usize, output_size: usize) -> Self;
    fn parameters(&self) -> Vec<&Tensor>;
    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor>;

}