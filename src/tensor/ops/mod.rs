// This file is used to manage backward operation of tensor.
// @Author: Yuyuan12138x@gmail.com

use super::Tensor;
use super::operation::Operation;
use anyhow::Result;
pub mod add;
pub mod div;
pub mod log;
pub mod matmul;
pub mod mean;
pub mod multiply;
pub mod pow;
pub mod relu;
pub mod sigmoid;
pub mod sub;
pub mod tanh;
pub mod softmax;
pub mod sum;
pub mod t;
pub mod unsqueeze;
pub mod squeeze;

pub fn _backward(tensor: &Tensor) -> Result<()>{
    let data = tensor.data.borrow();
    match data.operation {
        Operation::Add => add::backward(&tensor)?,
        Operation::Sub => sub::backward(&tensor)?,
        Operation::Multiply => multiply::backward(&tensor)?,
        Operation::Div => div::backward(&tensor).expect("Div backward went wrong"),
        Operation::Sigmoid => sigmoid::backward(&tensor)?,
        Operation::ReLU => relu::backward(&tensor)?,
        Operation::Matmul => matmul::backward(&tensor)?,
        Operation::Mean => mean::backward(&tensor)?,
        Operation::Log(base) => log::backward(&tensor, base)?,
        Operation::Pow(exponent) => pow::backward(&tensor, exponent)?,
        Operation::Tanh => tanh::backward(&tensor)?,
        Operation::Softmax => softmax::backward(&tensor)?,
        Operation::Sum => sum::backward(&tensor)?,
        Operation::T => t::backward(&tensor)?,
        Operation::Unsqueeze(dim) => unsqueeze::backward(&tensor, dim)?,
        Operation::Squeeze(dim) => squeeze::backward(&tensor, dim)?,

        Operation::None => {}
        _ => {
            panic!("No definitive operation {:?}", data.operation);
        }
    };

    Ok(())
}