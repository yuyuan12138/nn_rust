use std::cell::RefCell;
use std::rc::Rc;
use ndarray::{ArrayD, IxDyn};

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<RefCell<NodeData>>,
}

#[derive(Debug, Clone)]
pub enum TensorValue {
    Scalar(f64),
    Vector1D(Vec<f64>),
    Matrix2D(Vec<Vec<f64>>),
}

#[derive(Clone)]
pub struct NodeData {
    pub value: TensorValue,
    pub grad: TensorValue,
    pub operation: Operation,
    pub dependencies: Vec<Tensor>,
}

impl NodeData {
    fn add_grad_scalar(&mut self, delta: f64) {
        match &mut self.grad {
            TensorValue::Scalar(v) => *v += delta,
            TensorValue::Vector1D(vec) => vec.iter_mut().for_each(|x| *x += delta),
            TensorValue::Matrix2D(mat) => mat.iter_mut()
                .for_each(|row| row.iter_mut().for_each(|x| *x += delta)),
        }
    }

    fn add_grad(&mut self, delta: TensorValue) {
        match (&mut self.grad, delta) {
            (TensorValue::Scalar(a), TensorValue::Scalar(b)) => *a += b,
            (TensorValue::Vector1D(a), TensorValue::Vector1D(b)) => {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b)
            }
            (TensorValue::Matrix2D(a), TensorValue::Matrix2D(b)) => {
                a.iter_mut().zip(b).for_each(|(a_row, b_row)| {
                    a_row.iter_mut().zip(b_row).for_each(|(a, b)| *a += b)
                })
            }
            _ => panic!("Gradient shape mismatch"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Operation{
    None,
    Add,
    Sub,
    Multiply,
    Div,
    Matmul,
    Sigmoid,
    // TODO
}

impl TensorValue {
    fn zeros_like(&self) -> Self {
        match self {
            TensorValue::Scalar(_) => TensorValue::Scalar(0.0),
            TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![0.0; v.len()]),
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(vec![vec![0.0; m[0].len()]; m.len()])
            }
        }
    }

    fn shape(&self) -> Vec<usize> {
        match self {
            TensorValue::Scalar(_) => vec![],
            TensorValue::Vector1D(v) => vec![v.len()],
            TensorValue::Matrix2D(m) => vec![m.len(), m[0].len()],
        }
    }

    fn broadcast_mul(&self, other: &Self) -> Self {
        match (self, other) {
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s * x).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s * x).collect()).collect()
                )
            }
            _ => panic!("Unsupported broadcast combination"),
        }
    }

    fn sum(&self) -> f64 {
        match self {
            TensorValue::Scalar(s) => *s,
            TensorValue::Vector1D(v) => v.iter().sum(),
            TensorValue::Matrix2D(m) => m.iter().flat_map(|row| row.iter()).sum(),
        }
    }

    pub(crate) fn sub(&self, other: &Self) -> Self {
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
            _ => panic!("Mismatched types in sub operation"),
        }
    }

    fn scaled(&self, factor: f64) -> Self {
        match self {
            TensorValue::Scalar(v) => TensorValue::Scalar(v * factor),
            TensorValue::Vector1D(v) => {
                TensorValue::Vector1D(v.iter().map(|x| x * factor).collect())
            }
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row|
                        row.iter().map(|x| x * factor).collect()
                    ).collect()
                )
            }
        }
    }
}

impl Tensor {

    fn from_value(value: TensorValue) -> Self {
        let grad = match &value {
            TensorValue::Scalar(_) => TensorValue::Scalar(0.0),
            TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![0.0; v.len()]),
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(vec![vec![0.0; m[0].len()]; m.len()])
            }
        };

        Tensor {
            data: Rc::new(RefCell::new(NodeData {
                value,
                grad,
                operation: Operation::None,
                dependencies: vec![],
            })),
        }
    }

    pub fn scalar(value: f64) -> Self {
        Self::from_value(TensorValue::Scalar(value))
    }

    pub fn vector(data: Vec<f64>) -> Self {
        Self::from_value(TensorValue::Vector1D(data))
    }

    pub fn matrix(data: Vec<Vec<f64>>) -> Self {
        Self::from_value(TensorValue::Matrix2D(data))
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相加
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val + b_val)
            }
            // 向量相加
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a + b).collect()
                )
            }
            // 矩阵相加
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a + b).collect()
                    }).collect()
                )
            }
            // 标量广播
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| x + s).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x + s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x + s).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x + s).collect()).collect()
                )
            }
            _ => panic!("Invalid add operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Add;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相减
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val - b_val)
            }
            // 向量相减
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a - b).collect()
                )
            }
            // 矩阵相减
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a - b).collect()
                    }).collect()
                )
            }
            // 标量广播
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s - x).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x - s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s - x).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x - s).collect()).collect()
                )
            }
            _ => panic!("Invalid sub operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Sub;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn multiply(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相乘
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val * b_val)
            }
            // 向量相乘
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a * b).collect()
                )
            }
            // 矩阵相乘（逐元素）
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a * b).collect()
                    }).collect()
                )
            }
            // 标量广播
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s * x).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x * s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s * x).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x * s).collect()).collect()
                )
            }
            _ => panic!("Invalid multiply operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Multiply;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相除
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val / b_val)
            }
            // 向量相除
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a / b).collect()
                )
            }
            // 矩阵相除（逐元素）
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a / b).collect()
                    }).collect()
                )
            }
            // 标量广播
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s / x).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x / s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s / x).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x / s).collect()).collect()
                )
            }
            _ => panic!("Invalid div operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Div;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat[0].len(), b_mat.len(),
                           "Matrix multiplication dimension mismatch: {} vs {}",
                           a_mat[0].len(), b_mat.len());

                let mut result = vec![vec![0.0; b_mat[0].len()]; a_mat.len()];
                for i in 0..a_mat.len() {
                    for j in 0..b_mat[0].len() {
                        for k in 0..a_mat[0].len() {
                            result[i][j] += a_mat[i][k] * b_mat[k][j];
                        }
                    }
                }
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

    pub fn sigmoid(&self) -> Tensor {
        let a = self.data.borrow();

        let result_value = match &a.value {
            TensorValue::Scalar(v) => {
                let s = 1.0 / (1.0 + (-v).exp());
                TensorValue::Scalar(s)
            }
            TensorValue::Vector1D(v) => {
                TensorValue::Vector1D(
                    v.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
                )
            }
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| {
                        row.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
                    }).collect()
                )
            }
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Sigmoid;
            res_data.dependencies = vec![self.clone()];
        }
        result
    }

    pub fn backward(&self) {
        {
            let mut data = self.data.borrow_mut();
            data.grad = match &data.value {
                TensorValue::Scalar(_) => TensorValue::Scalar(1.0),
                TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![1.0; v.len()]),
                TensorValue::Matrix2D(m ) => {
                    TensorValue::Matrix2D(vec![vec![1.0; m[0].len()]; m.len()])
                }
            };
        }

        let mut nodes = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self._build_topo(&mut nodes, &mut visited);
        for node in nodes.iter().rev() {
            node._backward();
        }
    }

    fn _build_topo(&self, nodes: &mut Vec<Tensor>, visited: &mut std::collections::HashSet<*const NodeData>){
        let ptr = self.data.as_ptr() as *const NodeData;
        if visited.contains(&ptr){
            return ;
        }
        visited.insert(ptr);

        for dep in &self.data.borrow().dependencies {
            dep._build_topo(nodes, visited);
        }

        nodes.push(self.clone());
    }

    fn _backward(&self) {
        let data = self.data.borrow();
        match data.operation {
            Operation::Add => {
                let dependencies = &data.dependencies;

                if dependencies.len() != 2 {
                    panic!("Add operation requires exactly 2 dependencies");
                }
                let a = &dependencies[0];
                let b = &dependencies[1];

                match &data.grad {
                    TensorValue::Scalar(grad) => {
                        a.data.borrow_mut().add_grad_scalar(*grad);
                        b.data.borrow_mut().add_grad_scalar(*grad);
                    }
                    TensorValue::Vector1D(grad_vec) => {
                        a.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_vec.clone()));
                        b.data.borrow_mut().add_grad(TensorValue::Vector1D(
                            grad_vec.iter().map(|g| 1.0 * g).collect()
                        ));
                    }
                    TensorValue::Matrix2D(grad_mat) => {
                        a.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_mat.clone()));
                        b.data.borrow_mut().add_grad(TensorValue::Matrix2D(
                            grad_mat.iter().map(|row|
                                row.iter().map(|g| 1.0 * g).collect()
                            ).collect()
                        ));
                    }
                }
            }
            Operation::Sub => {
                let dependencies = &data.dependencies;
                if dependencies.len() != 2 {
                    panic!("Sub operation requires exactly 2 dependencies");
                }

                let a = &dependencies[0];
                let b = &dependencies[1];

                match &data.grad {
                    TensorValue::Scalar(grad) => {
                        a.data.borrow_mut().add_grad_scalar(*grad);
                        b.data.borrow_mut().add_grad_scalar(-*grad);
                    }
                    TensorValue::Vector1D(grad_vec) => {
                        a.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_vec.clone()));
                        b.data.borrow_mut().add_grad(TensorValue::Vector1D(
                            grad_vec.iter().map(|g| -g).collect()
                        ));
                    }
                    TensorValue::Matrix2D(grad_mat) => {
                        a.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_mat.clone()));
                        b.data.borrow_mut().add_grad(TensorValue::Matrix2D(
                            grad_mat.iter().map(|row|
                                row.iter().map(|g| -g).collect()
                            ).collect()
                        ));
                    }
                }
            }

            Operation::Multiply => {
                let dependencies = &data.dependencies;
                if dependencies.len() != 2 {
                    panic!("Multiply operation requires exactly 2 dependencies");
                }

                let a = &dependencies[0];
                let b = &dependencies[1];

                let (a_val, b_val) = {
                    let a_data = a.data.borrow();
                    let b_data = b.data.borrow();
                    (a_data.value.clone(), b_data.value.clone())
                };

                match (&data.grad, &a_val, &b_val) {
                    (TensorValue::Scalar(grad), TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                        a.data.borrow_mut().add_grad_scalar(grad * b_val);
                        b.data.borrow_mut().add_grad_scalar(grad * a_val);
                    }
                    (TensorValue::Vector1D(grad), TensorValue::Vector1D(a_val), TensorValue::Vector1D(b_val)) => {
                        let a_grad: Vec<_> = grad.iter().zip(b_val).map(|(g, b)| g * b).collect();
                        let b_grad: Vec<_> = grad.iter().zip(a_val).map(|(g, a)| g * a).collect();
                        a.data.borrow_mut().add_grad(TensorValue::Vector1D(a_grad));
                        b.data.borrow_mut().add_grad(TensorValue::Vector1D(b_grad));
                    }
                    (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(a_val), TensorValue::Matrix2D(b_val)) => {
                        let mut a_grad = vec![vec![0.0; a_val[0].len()]; a_val.len()];
                        let mut b_grad = vec![vec![0.0; b_val[0].len()]; b_val.len()];

                        for i in 0..grad.len() {
                            for j in 0..grad[0].len() {
                                a_grad[i][j] += grad[i][j] * b_val[i][j];
                                b_grad[i][j] += grad[i][j] * a_val[i][j];
                            }
                        }

                        a.data.borrow_mut().add_grad(TensorValue::Matrix2D(a_grad));
                        b.data.borrow_mut().add_grad(TensorValue::Matrix2D(b_grad));
                    }
                    _ => panic!("Unsupported multiply backward combination"),
                }
            }

            Operation::Div => {
                let dependencies = &data.dependencies;
                if dependencies.len() != 2 {
                    panic!("Div operation requires exactly 2 dependencies");
                }

                let a = &dependencies[0];
                let b = &dependencies[1];

                let (a_val, b_val) = {
                    let a_data = a.data.borrow();
                    let b_data = b.data.borrow();
                    (a_data.value.clone(), b_data.value.clone())
                };

                match (&data.grad, &a_val, &b_val) {
                    (TensorValue::Scalar(grad), TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                        let b_sq = b_val.powi(2);
                        a.data.borrow_mut().add_grad_scalar(grad / b_val);
                        b.data.borrow_mut().add_grad_scalar(-grad * a_val / b_sq);
                    }
                    // TODO: 其他类型处理类似，需要实现对应的梯度计算
                    _ => unimplemented!("Div backward for non-scalar not implemented"),
                }
            }

            Operation::Sigmoid => {
                let dependencies = &data.dependencies;
                if dependencies.len() != 1 {
                    panic!("Sigmoid operation requires exactly 1 dependency");
                }

                let x = &dependencies[0];
                let sigmoid_output = match &data.value {
                    TensorValue::Scalar(s) => TensorValue::Scalar(*s),
                    TensorValue::Vector1D(v) => TensorValue::Vector1D(v.clone()),
                    TensorValue::Matrix2D(m) => TensorValue::Matrix2D(m.clone()),
                };

                match (&data.grad, &sigmoid_output) {
                    (TensorValue::Scalar(grad), TensorValue::Scalar(s)) => {
                        let grad_x = grad * s * (1.0 - s);
                        x.data.borrow_mut().add_grad_scalar(grad_x);
                    }
                    (TensorValue::Vector1D(grad), TensorValue::Vector1D(s)) => {
                        let grad_x: Vec<_> = grad.iter()
                            .zip(s)
                            .map(|(g, s)| g * s * (1.0 - s))
                            .collect();
                        x.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_x));
                    }
                    (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(s)) => {
                        let grad_x: Vec<Vec<_>> = grad.iter()
                            .zip(s)
                            .map(|(g_row, s_row)| {
                                g_row.iter()
                                    .zip(s_row)
                                    .map(|(g, s)| g * s * (1.0 - s))
                                    .collect()
                            })
                            .collect();
                        x.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_x));
                    }
                    _ => panic!("Invalid sigmoid gradient combination"),
                }
            }

            Operation::Matmul => {
                todo!()
            }

            // 其他操作...
            _ => {}
        }
    }
}
