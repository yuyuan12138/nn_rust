use crate::tensor::Tensor;

pub fn mse_loss(predictions: Tensor, targets: Tensor) -> Tensor {
    // MSE = 1 / n * sum((y_i - y_hat_i) ^ 2)
    let diff = predictions.sub(&targets);
    let squared = diff.pow(2.0);
    squared.mean()
}

pub fn cross_entropy_loss(prediction: &[Tensor], targets: &[Tensor]) -> Tensor{
    todo!()
}