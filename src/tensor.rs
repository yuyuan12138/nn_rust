use std::cell::RefCell;
use std::rc::Rc;
use ndarray::{ArrayD, IxDyn};

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<RefCell<NodeData>>,
}


pub struct NodeData{
    pub value: ArrayD<f64>,
    pub grad: ArrayD<f64>,
    pub operation: Operation,
    pub dependencies: Vec<Tensor>,
}

#[derive(Clone, Copy)]
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

impl Tensor {

    pub fn scalar(value: f64) -> Self {

        Tensor {
            data: Rc::new(RefCell::new(NodeData {
                value: ArrayD::from_elem(IxDyn(&[]), value),
                grad: ArrayD::zeros(IxDyn(&[])),
                operation: Operation::None,
                dependencies: vec![],
            })),
        }
    }

    pub fn from_array(arr: ArrayD<f64>) -> Self {
        Tensor {
            data: Rc::new(RefCell::new(NodeData {
                value: arr,
                grad: ArrayD::zeros(IxDyn(&[])),
                operation: Operation::None,
                dependencies: vec![],
            })),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor{
        let a_data = self.data.borrow();
        let b_data = other.data.borrow();
        let a_val = &a_data.value;
        let b_val = &b_data.value;

        assert!(
            a_val.shape() == b_val.shape() ||
            a_val.ndim() == 0 ||
            b_val.ndim() == 0,
            "Shape does not match: {:?} vs {:?}",
            a_val.shape(),
            b_val.shape()
        );

        let result_val = a_val + b_val;
        let result = Tensor::from_array(result_val);

        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Add;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }

        result


    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a_data = self.data.borrow();
        let b_data = other.data.borrow();
        let a_val = &a_data.value;
        let b_val = &b_data.value;

        assert!(
            a_val.shape() == b_val.shape() ||
                a_val.ndim() == 0 ||
                b_val.ndim() == 0,
            "Shape does not match: {:?} vs {:?}",
            a_val.shape(),
            b_val.shape()
        );

        let result_val = a_val - b_val;
        let result = Tensor::from_array(result_val);

        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Sub;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }

        result
    }

    pub fn multiply(&self, other: &Tensor) -> Tensor {

        let a_data = self.data.borrow();
        let b_data = other.data.borrow();
        let a_val = &a_data.value;
        let b_val = &b_data.value;

        assert!(
            a_val.shape() == b_val.shape() ||
                a_val.ndim() == 0 ||
                b_val.ndim() == 0,
            "Shape does not match: {:?} vs {:?}",
            a_val.shape(),
            b_val.shape()
        );
        let result_val = a_val * b_val;
        let result = Tensor::from_array(result_val);

        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Multiply;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let a_data = self.data.borrow();
        let b_data = other.data.borrow();
        let a_val = &a_data.value;
        let b_val = &b_data.value;

        assert!(
            a_val.shape() == b_val.shape() ||
                a_val.ndim() == 0 ||
                b_val.ndim() == 0,
            "Shape does not match: {:?} vs {:?}",
            a_val.shape(),
            b_val.shape()
        );

        let result_val = a_val / b_val;
        let result = Tensor::from_array(result_val);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Div;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a_data = self.data.borrow();
        let b_data = other.data.borrow();
        let a_val = &a_data.value;
        let b_val = &b_data.value;
        assert!(
            a_val.shape().len() >= 2 && b_val.shape().len() >= 2,
            "Must >= 2 dims"
        );
        assert_eq!(
            a_val.shape()[a_val.ndim() - 1],
            b_val.shape()[b_val.ndim() - 2],
            "dim does not match: {:?} vs {:?}",
            a_val.shape(),
            b_val.shape(),
        );

        let a_2d = a_val.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let b_2d = b_val.clone().into_dimensionality::<ndarray::Ix2>().unwrap();

        let result_val = a_2d.dot(&b_2d).into_dyn();
        let result = Tensor::from_array(result_val);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Matmul;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }

        result

    }

    pub fn sigmoid(&self) -> Tensor {
        let a_data = self.data.borrow();

        let a_val = &a_data.value;

        let result_val = 1.0 / (1.0 + (-a_val).exp());
        let result = Tensor::from_array(result_val);

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
            data.grad = ArrayD::ones(data.value.shape());
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

    fn _backward(&self){
        let data = self.data.borrow();
        match data.operation{
            Operation::Add => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                let grad = &data.grad;

                let a_grad = grad.clone();
                let b_grad = grad.clone();
                a.data.borrow_mut().grad += &a_grad;
                b.data.borrow_mut().grad += &b_grad;
            },

            Operation::Sub => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                let grad = &data.grad;

                let a_grad = grad.clone();
                let b_grad = grad.clone();

                a.data.borrow_mut().grad += &a_grad;
                b.data.borrow_mut().grad -= &b_grad;
            }

            Operation::Multiply => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                let grad = &data.grad;
                let (a_val, b_val) = {
                    let a_data = a.data.borrow();
                    let b_data = b.data.borrow();
                    (a_data.value.clone(), b_data.value.clone())
                };

                let a_grad = grad * &b_val;
                let b_grad = grad * &a_val;


                a.data.borrow_mut().grad += &a_grad;
                b.data.borrow_mut().grad += &b_grad;
            },

            Operation::Div => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                let grad = &data.grad;

                let (a_val, b_val) = {
                    let a_data = a.data.borrow();
                    let b_data = b.data.borrow();
                    (a_data.value.clone(), b_data.value.clone())
                };


                let a_grad = grad * (1.0 / &b_val);
                let b_grad = grad * (&a_val / (&b_val.powf(2.0)));


                a.data.borrow_mut().grad += &a_grad;
                b.data.borrow_mut().grad += &b_grad;
            },

            Operation::Matmul => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                let grad = &data.grad;

                let (a_val, b_val) = {
                    let a_data = a.data.borrow();
                    let b_data = b.data.borrow();
                    (a_data.value.clone(), b_data.value.clone())
                };
                let a_2d = a_val.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let b_2d = b_val.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                let grad_2d = grad.clone().into_dimensionality::<ndarray::Ix2>().unwrap();

                let a_grad = grad_2d.dot(&b_2d.t());
                let b_grad = a_2d.t().dot(&grad_2d);

                {
                    let mut a_data = a.data.borrow_mut();
                    a_data.grad += &a_grad;
                }
                {
                    let mut b_data = b.data.borrow_mut();
                    b_data.grad += &b_grad;
                }
            },

            Operation::Sigmoid => {
                let x = &data.dependencies[0];
                let grad = &data.grad;
                let val = {
                    let x_data = x.data.borrow();
                    x_data.value.clone()
                };

                let a_grad = grad * ((-val.exp()) / (1.0 + -val.exp()).powf(2.0));

                x.data.borrow_mut().grad += &a_grad;
            }

            Operation::None => {}
        }

        for dep in &data.dependencies {
            dep._backward();
        }
    }
}

// TODO: update to 2D or 3D tensor.