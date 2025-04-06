// #![recursion_limit = "4096"]
mod tensor;
mod nn;
mod loss_fn;

use ndarray::{ArrayD, IxDyn};
use tensor::Tensor;
use loss_fn::mse_loss;
use nn::Layer;
use nn::Optimizer;
use nn::optimizer::SGD;

fn test_XOR(){
    // TODO: an example about XOR function. [1, 0] || [0, 1] -> 1 [0, 0] || [1, 1] -> 0; DONE !
    let x1 = Tensor::scalar(1.0);
    let x2 = Tensor::scalar(0.0);

    let layer = nn::layer::Linear::new(2, 1);
    let layer2 = nn::layer::Linear::new(2, 2);
    let input_1 = Tensor::vector(vec![1.0, 1.0]);
    let input_2 = Tensor::vector(vec![0.0, 0.0]);
    let input_3 = Tensor::vector(vec![1.0, 0.0]);
    let input_4 = Tensor::vector(vec![0.0, 1.0]);

    let inputs = [&input_1, &input_2, &input_3, &input_4];
    // println!("{:?}", layer.parameters());
    let targets = [
        Tensor::vector(vec![0.0]),
        Tensor::vector(vec![0.0]),
        Tensor::vector(vec![1.0]),
        Tensor::vector(vec![1.0]),
    ];

    let optimizer = SGD::new(0.01);

    for _ in 0..10000 {
        for (i, input) in inputs.iter().enumerate() {
            optimizer.zero_grad(&layer.parameters());
            optimizer.zero_grad(&layer2.parameters());
            let outputs_2 = layer2.forward(input);
            let outputs = layer.forward(&outputs_2);

            let loss = mse_loss(outputs, targets[i].clone());
            loss.backward();
            optimizer.step(&layer.parameters());
            optimizer.step(&layer2.parameters());
        }
    }

    for (i, input) in inputs.iter().enumerate() {
        let outputs_2 = layer2.forward(input);
        let outputs = layer.forward(&outputs_2);
        // println!("{:?}", layer.parameters());
        println!("Updated output: {:?}", outputs.data.borrow().value);
    }
}


fn main() {
    test_XOR();

}
