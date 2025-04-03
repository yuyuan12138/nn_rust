use crate::tensor::Tensor;

pub struct DenseLayer {
    pub weights: Vec<Vec<Tensor>>,
    pub bias: Tensor,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            let mut w = Vec::with_capacity(input_size);
            for _ in 0..input_size {
                w.push(Tensor::new(rand::random::<f64>() * 0.1));
            }
            weights.push(w); // TODO: a simple example
        }

        DenseLayer {
            // TODO: Example
            weights,
            bias: Tensor::new(rand::random::<f64>() * 0.1),
        }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut outputs = Vec::new();
        for w in &self.weights {
            let mut sum = self.bias.clone();
            for (i, input) in inputs.iter().enumerate() {

                sum = sum.add(&input.multiply(&w[i]));
            }
            outputs.push(sum.sigmoid());
        }

        outputs
    }
}