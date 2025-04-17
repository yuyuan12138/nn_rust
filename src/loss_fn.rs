use crate::tensor::Tensor;
use anyhow::Result;

pub fn mse_loss(predictions: Tensor, targets: Tensor) -> Result<Tensor> {
    // MSE = 1 / n * sum((y_i - y_hat_i) ^ 2)
    let diff = predictions.sub(&targets)?;
    let squared = diff.pow(2.0)?;
    Ok(squared.mean()?)
}

pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let left = predictions.log(1.0_f64.exp())?.multiply(&targets)?;
    let one = Tensor::vector(vec![1.0]);
    let right = one.sub(&predictions)?.log(1.0_f64.exp())?.multiply(&one.sub(&targets)?)?;
    let negative_one = Tensor::scalar(-1.0);
    Ok(left.add(&right)?.mean()?.multiply(&negative_one)?)
}

pub fn cross_entropy_loss(prediction: &[Tensor], targets: &[Tensor]) -> Tensor{
    todo!()
}