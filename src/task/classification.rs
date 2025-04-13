use super::LearningTask;

use crate::{nn, tensor::Tensor, loss_fn::bce_loss};


pub struct BinaryClassificationTask {
    input_size: usize,
    hidden_size: usize,
}

impl BinaryClassificationTask {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self { input_size, hidden_size }
    }
}

impl LearningTask for BinaryClassificationTask {
    type Input = Tensor;
    type Output = f64;
    type Model = todo!();
}