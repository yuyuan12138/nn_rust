use crate::nn::dense::DenseLayer;

pub fn sgd_step(layer: &DenseLayer, learning_rate: f64) {
    for w in &layer.weights {
        for node in w {
            let mut data = node.data.borrow_mut();
            data.value -= learning_rate * data.grad;
            data.grad = 0.0;
        }
    }

    let mut bias_data = layer.bias.data.borrow_mut();
    bias_data.value -= learning_rate * bias_data.grad;
    bias_data.grad = 0.0;
}