use crate::tensor::Tensor;
use super::Layer;

pub struct Parameter1D {
    pub weights: Vec<Tensor>,
    pub bias: Tensor,
}

pub struct Parameter2D {
    pub weights: Vec<Vec<Tensor>>,
    pub bias: Tensor,
}

impl Layer for Parameter1D {
    fn new(input_size: usize, output_size: usize) -> Self {
        assert_eq!(input_size, output_size, "input_size and output_size must be the same in Parameter1D.");
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size{
            weights.push(Tensor::scalar(rand::random::<f64>() * 0.1));
        }

        Parameter1D{
            weights,
            bias: Tensor::scalar(rand::random::<f64>() * 0.1),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for row in &self.weights {
            params.push(row);
        }
        params.push(&self.bias);

        params
    }

    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut outputs = Vec::new();


        for (i, input) in inputs.iter().enumerate() {
            let mut sum = self.bias.clone();
            sum = sum.add(&input.multiply(&self.weights[i]));
            outputs.push(sum);
        }

        outputs
    }
}

impl Layer for Parameter2D {
    fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let mut w = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                w.push(Tensor::scalar(rand::random::<f64>() * 0.1));
            }
            weights.push(w);
        }

        Parameter2D{
            weights,
            bias: Tensor::scalar(rand::random::<f64>() * 0.1),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for row in &self.weights{
            for weight in row{
                params.push(weight);
            }
        }
        params.push(&self.bias);

        params
    }

    fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut outputs = Vec::new();
        for w in &self.weights {
            let mut sum = self.bias.clone();
            for (i, input) in inputs.iter().enumerate() {
                sum = sum.add(&input.multiply(&w[i]));
            }
            outputs.push(sum);
        }
        outputs
    }
}