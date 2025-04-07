mod tensor;
mod nn;
mod loss_fn;

use tensor::Tensor;
use nn::Layer;
use nn::Optimizer;
use nn::optimizer::SGD;
use crate::loss_fn::bce_loss;
use crate::tensor::value::TensorValue;

fn test_xor(){
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

    for j in 0..100_000 {
        for (_i, (input, target)) in inputs.iter().zip(&targets).enumerate() {
            optimizer.zero_grad(&layer1.parameters());
            optimizer.zero_grad(&layer2.parameters());
            // println!("Layer1 bias value origin: {:?}", layer1.params.weights.data.borrow().value);
            let hidden = layer1.forward(&input).tanh();
            let outputs = layer2.forward(&hidden).sigmoid();

            let loss = bce_loss(&outputs, &target);
            // println!("{:?}", outputs.data.borrow().value);
            // println!("{:?}", loss.data.borrow().value);
            loss.backward();
            // println!("Layer1 weights grad: {:?}", layer1.params.weights.data.borrow().grad);
            optimizer.step(&layer1.parameters());
            optimizer.step(&layer2.parameters());
            // println!("Layer1 bias value new: {:?}", layer1.params.weights.data.borrow().value);


        }
        // println!()
    }


    for (_i, (input, target)) in inputs.iter().zip(&targets).enumerate() {
        let hidden = layer1.forward(&input).tanh();
        let outputs = layer2.forward(&hidden).sigmoid();
        println!("Updated output: {:?}", outputs.data.borrow().value);
    }

}




fn main() {
    test_xor();
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rand_test(){
        let a = vec![rand::random::<f64>() * 0.1; 2];
        assert_eq!(a[0], a[1]);
    }

    #[test]
    fn add_works(){
        let a = Tensor::scalar(2.0);
        let b = Tensor::scalar(3.0);
        let c = a.add(&b);
        c.backward();
        let a_grad = match &a.data.borrow().grad {
            TensorValue::Scalar(s) => *s,
            _ => panic!("Error!")
        };
        let b_grad = match &b.data.borrow().grad {
            TensorValue::Scalar(s) => *s,
            _ => panic!("Error!")
        };
        assert_eq!(a_grad, 1.0);
        assert_eq!(b_grad, 1.0);
    }

    #[test]
    fn matmul_works(){
        let a = Tensor::matrix(vec![vec![2.0, 3.0], vec![3.0, 4.0]]);
        let b = Tensor::matrix(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let c = a.matmul(&b);
        c.backward();

        let a_grad = match &a.data.borrow().grad {
            TensorValue::Matrix2D(m) => m.clone(),
            _ => panic!("Error!")
        };
        let b_grad = match &b.data.borrow().grad {
            TensorValue::Matrix2D(m) => m.clone(),
            _ => panic!("Error!")
        };

        assert_eq!(a_grad, vec![vec![11.0, 15.0], vec![11.0, 15.0]]);
        assert_eq!(b_grad, vec![vec![5.0, 5.0], vec![7.0, 7.0]]);
    }
}
