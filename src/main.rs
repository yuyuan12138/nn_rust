mod tensor;
mod nn;
mod loss_fn;
mod optimizer;

use tensor::Tensor;
use nn::dense::DenseLayer;
use loss_fn::mse_loss;
use crate::optimizer::sgd_step;

fn main() {
    // TODO: an example about XOR function. [1, 0] || [0, 1] -> 1 [0, 0] || [1, 1] -> 0; DONE !
    let x1 = Tensor::new(1.0);
    let x2 = Tensor::new(0.0);

    let layer = DenseLayer::new(2, 1);
    let input_1 = vec![x1.clone(), x1.clone()];
    let input_2 = vec![x1.clone(), x2.clone()];
    let input_3 = vec![x2.clone(), x1.clone()];
    let input_4 = vec![x2.clone(), x2.clone()];

    let inputs = [&input_1, &input_2, &input_3, &input_4];

    let targets = [[x2.clone()], [x1.clone()], [x2.clone()], [x1.clone()]];

    let learning_rate = 3.0;
    let loss_fn = mse_loss;
    let optimizer = sgd_step;

    for _ in 0..100 {
        for (i, input) in inputs.iter().enumerate() {
            let outputs = layer.forward(*input);
            let loss = loss_fn(&outputs, &targets[i]);
            loss.backward();
            optimizer(&layer, learning_rate);
        }
    }

    for (i, input) in inputs.iter().enumerate() {
        let outputs = layer.forward(*input);
        let loss = loss_fn(&outputs, &targets[i]);
        loss.backward();
        optimizer(&layer, learning_rate);
        println!("Updated output: {:.4}", outputs[0].data.borrow().value);
    }


}
