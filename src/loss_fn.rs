use crate::tensor::Tensor;

pub fn mse_loss(prediction: &[Tensor], targets: &[Tensor]) -> Tensor {
    assert_eq!(prediction.len(), targets.len(), "Predictions and targets must have the same length.");
    let n = prediction.len() as f64;
    let mut total_loss = Tensor::new(0.0);
    for (pred, target) in prediction.iter().zip(targets.iter()){
        let diff = pred.sub(&target);
        let squared_diff = diff.multiply(&diff);
        total_loss = total_loss.add(&squared_diff);
    }
    total_loss.multiply(&Tensor::new(1.0 / n))
}

pub fn cross_entropy_loss(prediction: &[Tensor], targets: &[Tensor]) -> Tensor{
    todo!()
}