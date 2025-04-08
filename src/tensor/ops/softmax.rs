use crate::tensor::operation::Operation;
use crate::tensor::utils::transpose;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn softmax(&self) -> Tensor {
        let data = self.data.borrow();
        let result_value = match &data.value {
            TensorValue::Vector1D(v) => {
                let max = v.iter().fold(f64::MIN, |a, &b| a.max(b));
                let exps: Vec<f64> = v.iter().map(|x| (x - max).exp()).collect();

                let sum: f64 = exps.iter().sum();
                TensorValue::Vector1D(
                    exps.into_iter().map(|x| x / sum).collect()
                )
            }
            TensorValue::Matrix2D(m) => {

                assert!(!m.is_empty(), "Matrix is empty");
                let cols = m[0].len();
                assert!(
                    m.iter().all(|row| row.len() == cols),
                    "Inconsistent matrix columns"
                );

                let transposed = transpose(m);
                let processed_transposed: Vec<Vec<f64>> = transposed
                    .into_iter()
                    .map(|col| {
                        let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let exps: Vec<f64> = col.iter().map(|&x| (x - max).exp()).collect();

                        let sum_exps: f64 = exps.iter().sum();
                        exps.into_iter().map(|exp| exp / sum_exps).collect()
                    }).collect();

                let result = transpose(&processed_transposed);

                TensorValue::Matrix2D(result)
            }
            _ => panic!("softmax only supported for 1D/2D"),
        };
        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Softmax;
            res_data.dependencies = vec![self.clone()];
        }

        result
    }
}

pub fn backward(tensor: &Tensor) {
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Softmax backward requires exactly 1 dependency");
    }
    let input = &dependencies[0];

    let s = {
        let forward_data = tensor.data.borrow();
        match &forward_data.value {
            TensorValue::Vector1D(v) => TensorValue::Vector1D(v.clone()),
            TensorValue::Matrix2D(m) => TensorValue::Matrix2D(m.clone()),
            _ => panic!("Softmax output must be Vector1D or Matrix2D"),
        }
    };

    let dy = match &data.grad {
        TensorValue::Vector1D(grad) => TensorValue::Vector1D(grad.clone()),
        TensorValue::Matrix2D(grad) => TensorValue::Matrix2D(grad.clone()),
        _ => panic!("Softmax gradient must match output shape"),
    };

    let dz = match (&s, &dy) {
        (TensorValue::Vector1D(s_vec), TensorValue::Vector1D(dy_vec)) => {
            assert_eq!(s_vec.len(), dy_vec.len(), "Gradient shape mismatch");
            let sum_s_dy: f64 = s_vec.iter().zip(dy_vec).map(|(&s_i, &dy_i)| s_i * dy_i).sum();
            let dz_vec: Vec<f64> = s_vec.iter()
                .zip(dy_vec)
                .map(|(&s_i, &dy_i)| s_i * (dy_i - sum_s_dy))
                .collect();
            TensorValue::Vector1D(dz_vec)
        }
        (TensorValue::Matrix2D(s_mat), TensorValue::Matrix2D(dy_mat)) => {
            assert_eq!(s_mat.len(), dy_mat.len(), "Gradient rows mismatch");
            let cols = s_mat[0].len();

            // 转置以按列处理
            let s_transposed = transpose(s_mat);
            let dy_transposed = transpose(dy_mat);

            let dz_transposed: Vec<Vec<f64>> = s_transposed.into_iter()
                .zip(dy_transposed.into_iter())
                .map(|(s_col, dy_col)| {
                    let sum_s_dy: f64 = s_col.iter()
                        .zip(&dy_col)
                        .map(|(&s, &dy)| s * dy)
                        .sum();

                    s_col.into_iter()
                        .zip(dy_col.into_iter())
                        .map(|(s, dy)| s * (dy - sum_s_dy))
                        .collect()
                }).collect();

            // 转置回原始形状
            let dz_mat = transpose(&dz_transposed);
            TensorValue::Matrix2D(dz_mat)
        }
        _ => panic!("Softmax gradient and output type mismatch"),
    };
    input.data.borrow_mut().add_grad(dz);
}

#[test]
fn softmax_backward_works() {
    let input = Tensor::matrix(vec![
        vec![1.0, 1.0],
        vec![-1.0, 0.0]
    ]);

    let output = input.softmax().sum();

    output.backward();

    let input_grad = match &input.data.borrow().grad {
        TensorValue::Matrix2D(m) => m.clone(),
        _ => panic!("Input gradient should be Matrix2D")
    };


    let expected_grad = vec![
        vec![5.2500e-08, 0.0],
        vec![7.1050e-09, 0.0]
    ];


    for (row_actual, row_expected) in input_grad.iter().zip(expected_grad) {
        for (&val_actual, val_expected) in row_actual.iter().zip(row_expected) {
            approx::assert_abs_diff_eq!(
            val_actual,
            val_expected,
            epsilon = 1e-4
        );
        }
    }
}