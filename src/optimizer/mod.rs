use crate::tensor::Tensor;

pub mod sgd;

pub trait Optimizer {
    fn step(&self, params: &[&Tensor]);

    fn zero_grad(&self, params: &[&Tensor]);
}