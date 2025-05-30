mod tensor;
mod nn;
mod loss_fn;

use tensor::Tensor;
use nn::Layer;
use nn::Optimizer;
use nn::optimizer::SGD;
use crate::loss_fn::bce_loss;
use rayon::prelude::*;
use anyhow::Result;

fn test_xor() -> Result<()>{
    let layer1 = nn::layer::Linear::new(2, 4);
    let layer2 = nn::layer::Linear::new(4, 1);
    let inputs = vec![
        Tensor::vector(vec![1.0, 1.0]),
        Tensor::vector(vec![0.0, 1.0]),
        Tensor::vector(vec![1.0, 0.0]),
        Tensor::vector(vec![0.0, 0.0]),
    ];

    let targets = vec![
        Tensor::vector(vec![0.0]),
        Tensor::vector(vec![1.0]),
        Tensor::vector(vec![1.0]),
        Tensor::vector(vec![0.0])
    ];


    let optimizer = SGD::new(0.1);

    for _ in 0..100_000 {
        for (_i, (input, target)) in inputs.iter().zip(&targets).enumerate() {
            optimizer.zero_grad(&layer1.parameters());
            optimizer.zero_grad(&layer2.parameters());
            let hidden = layer1.forward(&input)?.tanh()?;
            let outputs = layer2.forward(&hidden)?.sigmoid()?;

            let loss = bce_loss(&outputs, &target)?;
            loss.backward()?;
            optimizer.step(&layer1.parameters());
            optimizer.step(&layer2.parameters());
        }
    }


    for (_i, (input, _target)) in inputs.iter().zip(&targets).enumerate() {
        let hidden = layer1.forward(&input)?.tanh()?;
        let outputs = layer2.forward(&hidden)?.sigmoid()?;
        println!("Updated output: {:?}", outputs.data.borrow().value);
    }

    Ok(())
}




fn main() -> Result<()>{
    test_xor()?;
    Ok(())
}


