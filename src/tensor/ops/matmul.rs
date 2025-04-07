use crate::tensor::operation::Operation;
use crate::tensor::utils::{matrix_multiply, transpose};
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            (TensorValue::Matrix2D(a_mat), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_mat[0].len(), b_vec.len(),
                           "Matrix multiplication dimension mismatch: {} vs {}",
                           a_mat[0].len(), b_vec.len()
                );
                let mut result = vec![0.0; a_mat.len()];
                for i in 0..a_mat.len() {
                    for j in 0..a_mat[0].len() {
                        result[i] += a_mat[i][j] * b_vec[j];
                    }
                }
                TensorValue::Vector1D(result)
            },
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat[0].len(), b_mat.len(),
                           "Matrix multiplication dimension mismatch: {} vs {}",
                           a_mat[0].len(), b_mat.len());

                let result = matrix_multiply(a_mat, b_mat);
                TensorValue::Matrix2D(result)
            }
            _ => panic!("Matmul only supported for 2D matrices"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Matmul;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }
}

pub fn backward(tensor: &Tensor){
    let data = tensor.data.borrow();

    // TODO
    let dependencies = &data.dependencies;
    if dependencies.len() != 2 {
        panic!("Matmul operation requires exactly 2 dependencies");
    }

    let a = &dependencies[0];
    let b = &dependencies[1];

    let grad = &data.grad;

    let (a_val, b_val) = {
        let a_data = a.data.borrow();
        let b_data = b.data.borrow();
        (a_data.value.clone(), b_data.value.clone())
    };

    match (&a_val, &b_val) {
        (TensorValue::Matrix2D(a_mat), TensorValue::Vector1D(b_vec)) => {
            let (m, n) = (a_mat.len(), a_mat[0].len());
            let p = 1;

            let grad_mat = match grad {
                TensorValue::Vector1D(v) => {
                    assert_eq!(
                        v.len(), m,
                        "Gradient vector length mismatch: expected {}, got {}",
                        m, v.len()
                    );
                    v.iter().map(|&x| vec![x]).collect::<Vec<Vec<f64>>>()
                },
                TensorValue::Matrix2D(mat) => {
                    assert_eq!(
                        mat.len(), m,
                        "Gradient matrix row mismatch: expected {}, got {}",
                        m, mat.len()
                    );
                    assert_eq!(
                        mat[0].len(), p,
                        "Gradient matrix should have 1 column, got {}",
                        mat[0].len()
                    );
                    mat.clone()
                },
                _ => panic!("Invalid gradient shape for matrix-vector matmul")
            };

            let b_row = vec![b_vec.clone()]; // 1 x n
            let da = matrix_multiply(&grad_mat, &b_row);

            let a_t = transpose(a_mat);
            let db_mat = matrix_multiply(&a_t, &grad_mat);

            let db = db_mat.into_iter()
                .map(|row| row[0])
                .collect::<Vec<f64>>();

            assert_eq!(da.len(), m, "dA row mismatch");
            assert_eq!(da[0].len(), n, "dA column mismatch");
            assert_eq!(db.len(), n, "dB length mismatch");

            a.data.borrow_mut().add_grad(TensorValue::Matrix2D(da));
            b.data.borrow_mut().add_grad(TensorValue::Vector1D(db));

        }
        (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
            let (m, n) = (a_mat.len(), a_mat[0].len());
            let (n_, p) = (b_mat.len(), b_mat[0].len());

            let grad_ = match &data.grad {
                TensorValue::Matrix2D(grad_) => grad_,
                _ => panic!("")
            };
            assert_eq!(n, n_, "Matmul dimension mismatch: A.cols({}) != B.rows({})", n, n_);
            assert_eq!(grad_.len(), m, "Gradient rows mismatch: excepted {}, got {}", m, grad_.len());
            assert_eq!(grad_[0].len(), p, "Gradient rows mismatch: excepted {}, got {}", p, grad_.len());

            let b_t = transpose(b_mat);
            let da = matrix_multiply(&grad_, &b_t);

            let a_t = transpose(a_mat);
            let db = matrix_multiply(&a_t, &grad_);

            a.data.borrow_mut().add_grad(TensorValue::Matrix2D(da));
            b.data.borrow_mut().add_grad(TensorValue::Matrix2D(db));
        }

        _ => panic!("Unsupported Matmul backward combination"),
    }
}