use crate::tensor::Tensor;

pub fn mse_loss(predictions: Tensor, targets: Tensor) -> Tensor {
    // MSE = 1 / n * sum((y_i - y_hat_i) ^ 2)
    let diff = predictions.sub(&targets);
    let squared = diff.pow(2.0);
    squared.mean()
}

pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let left = predictions.log(1.0_f64.exp()).multiply(&targets);
    let one = Tensor::vector(vec![1.0]);
    let right = one.sub(&predictions).log(1.0_f64.exp()).multiply(&one.sub(&targets));
    let negative_one = Tensor::vector(vec![-1.0]);
    left.add(&right).multiply(&negative_one)
}

pub fn cross_entropy_loss(prediction: &[Tensor], targets: &[Tensor]) -> Tensor{
    todo!()
}