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
use crate::tensor::TensorValue;

fn test_XOR(){
    // TODO: an example about XOR function. [1, 0] || [0, 1] -> 1 [0, 0] || [1, 1] -> 0; DONE !
    let x1 = Tensor::scalar(1.0);
    let x2 = Tensor::scalar(0.0);

    let layer = nn::layer::Linear::new(2, 1);
    let input_1 = Tensor::vector(vec![1.0, 1.0]);
    let input_2 = Tensor::vector(vec![0.0, 0.0]);
    let input_3 = Tensor::vector(vec![1.0, 0.0]);
    let input_4 = Tensor::vector(vec![0.0, 1.0]);

    let inputs = [&input_1, &input_2, &input_3, &input_4];

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
            let outputs = layer.forward(&input);

            let loss = mse_loss(outputs, targets[i].clone());
            loss.backward();
            optimizer.step(&layer.parameters());

        }
    }

    for (i, input) in inputs.iter().enumerate() {
        let outputs = layer.forward(&input);
        // println!("outputs_2: {:?}", outputs_2.data.borrow().value);
        println!("Updated output: {:?}", outputs.data.borrow().value);
    }
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
