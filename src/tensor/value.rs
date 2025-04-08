#[derive(Debug, Clone)]
pub enum TensorValue {
    Scalar(f64),
    Vector1D(Vec<f64>),
    Matrix2D(Vec<Vec<f64>>),
    Tensor3D(Vec<Vec<Vec<f64>>>),
}

impl TensorValue {
    fn zeros_like(&self) -> Self {
        match self {
            TensorValue::Scalar(_) => TensorValue::Scalar(0.0),
            TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![0.0; v.len()]),
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(vec![vec![0.0; m[0].len()]; m.len()])
            }
            TensorValue::Tensor3D(t) => {
                TensorValue::Tensor3D(vec![vec![vec![0.0; t[0][0].len()]; t[0].len()]; t.len()])
            }
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        match self {
            TensorValue::Scalar(_) => vec![],
            TensorValue::Vector1D(v) => vec![v.len()],
            TensorValue::Matrix2D(m) => vec![m.len(), m[0].len()],
            TensorValue::Tensor3D(t) => vec![t.len(), t[0].len(), t[0][0].len()],
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (TensorValue::Scalar(a), TensorValue::Scalar(b)) => TensorValue::Scalar(a - b),
            (TensorValue::Vector1D(a), TensorValue::Vector1D(b)) => {
                assert_eq!(a.len(), b.len(), "Vector length mismatch in sub");
                TensorValue::Vector1D(
                    a.iter().zip(b).map(|(x, y)| x - y).collect()
                )
            }
            (TensorValue::Matrix2D(a), TensorValue::Matrix2D(b)) => {
                assert_eq!(a.len(), b.len(), "Matrix row mismatch in sub");
                assert_eq!(a[0].len(), b[0].len(), "Matrix column mismatch in sub");
                TensorValue::Matrix2D(
                    a.iter().zip(b).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(x, y)| x - y).collect()
                    }).collect()
                )
            }
            (TensorValue::Tensor3D(a), TensorValue::Tensor3D(b)) => {
                assert_eq!(a.len(), b.len(), "Matrix row mismatch in sub");
                assert_eq!(a[0].len(), b[0].len(), "Matrix column mismatch in sub");
                assert_eq!(a[0][0].len(), b[0][0].len(), "Matrix column mismatch in sub");

                TensorValue::Tensor3D(
                    a.iter().zip(b).map(|(a_1, b_1)| {
                        a_1.iter().zip(b_1).map(|(a_2, b_2)| {
                            a_2.iter().zip(b_2).map(|(a_3, b_3)| a_3 - b_3).collect()
                        }).collect()
                    }).collect()
                )
            }
            _ => panic!("Mismatched types in sub operation"),
        }
    }
}