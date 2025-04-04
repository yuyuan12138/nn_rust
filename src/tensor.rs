use std::cell::RefCell;
use std::rc::Rc;
use ndarray::{Array, ArrayD, IxDyn};

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
            data.grad = ArrayD::from_elem(IxDyn(&[]), 1.0);
        }
        self._backward();
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