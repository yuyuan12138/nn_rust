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
use crate::loss_fn::bce_loss;
use crate::tensor::value::TensorValue;

fn test_XOR(){


    let layer1 = nn::layer::Linear::new(2, 4);
    let layer2 = nn::layer::Linear::new(4, 1);
    let inputs = Tensor::matrix(vec![vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 0.0, 1.0, 0.0]]);



    let targets = Tensor::matrix(vec![vec![
        0.0,
        1.0,
        0.0,
        1.0,
    ]]);

    let optimizer = SGD::new(0.01);

    for _ in 0..1000_000 {

        optimizer.zero_grad(&layer1.parameters());
        optimizer.zero_grad(&layer2.parameters());
        let hidden = layer1.forward(&inputs);

        let outputs = layer2.forward(&hidden);
        let loss = bce_loss(outputs, targets.clone());
        loss.backward();
        optimizer.step(&layer1.parameters());
        optimizer.step(&layer2.parameters());
    }


    let hidden = layer1.forward(&inputs);
    let outputs = layer2.forward(&hidden);

    println!("Updated output: {:?}", outputs.data.borrow().value);

}




fn main() {
    test_XOR();
    // add()
}

fn add(){

    let a = Tensor::matrix(vec![vec![2.0, 3.0]]);
    let b = Tensor::vector(vec![4.0, 5.0]);
    let output = a.matmul(&b);
    output.backward();
    println!("{:?}", a);
    {
        let mut data = a.data.borrow_mut();

        let update = match &data.grad {
            TensorValue::Scalar(g) => TensorValue::Scalar(1.0 * g),
            TensorValue::Vector1D(v) => {
                TensorValue::Vector1D(v.iter().map(|g| 1.0 * g).collect())
            }
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row|
                        row.iter().map(|g| 1.0 * g).collect()
                    ).collect()
                )
            }
        };

        data.value = data.value.sub(&update);
    }
    println!("{:?}", a);
    println!();
    println!("{:?}", b);
}
