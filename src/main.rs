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
    let input_1 = vec![x1.clone(), x1.clone()];
    let input_2 = vec![x1.clone(), x2.clone()];
    let input_3 = vec![x2.clone(), x1.clone()];
    let input_4 = vec![x2.clone(), x2.clone()];

    let inputs = [&input_1, &input_2, &input_3, &input_4];

    let targets = [[x2.clone()], [x1.clone()], [x2.clone()], [x1.clone()]];

    let loss_fn = mse_loss;
    let optimizer = SGD::new(0.00001);

    for _ in 0..100000 {
        for (i, input) in inputs.iter().enumerate() {
            optimizer.zero_grad(&layer.parameters());
            let outputs = layer.forward(input);
            let loss = loss_fn(&outputs, &targets[i]);
            loss.backward();
            optimizer.step(&layer.parameters());
        }
    }

    for (i, input) in inputs.iter().enumerate() {
        optimizer.zero_grad(&layer.parameters());
        let outputs = layer.forward(input);
        let loss = loss_fn(&outputs, &targets[i]);
        loss.backward();
        optimizer.step(&layer.parameters());
        println!("Updated output: {:.4}", outputs[0].data.borrow().value);
    }
}


fn main() {
    test_XOR();
    let a = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap());
    let b = Tensor::from_array(ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).unwrap());

    let c = a.matmul(&b);

    println!("Result:\n {:?}", c.data.borrow().value);

}
