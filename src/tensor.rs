use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone)]
pub struct Tensor {
    pub data: Rc<RefCell<NodeData>>,
}


pub struct NodeData{
    pub value: f64,
    pub grad: f64,
    pub operation: Operation,
    pub dependencies: Vec<Tensor>,
}

#[derive(Clone, Copy)]
pub enum Operation{
    None,
    Add,
    Sub,
    Multiply,
    Sigmoid,
    // TODO
}

impl Tensor {
    pub fn new(value: f64) -> Self {
        Tensor {
            data: Rc::new(RefCell::new(NodeData {
                value,
                grad: 0.0,
                operation: Operation::None,
                dependencies: vec![],
            })),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor{
        let (a_val, b_val) = (
            self.data.borrow().value,
            other.data.borrow().value
            );
        let result = Tensor::new(a_val + b_val);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Add;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let (a_val, b_val) = (
            self.data.borrow().value,
            other.data.borrow().value
            );
        let result = Tensor::new(a_val - b_val);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Sub;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }

        result
    }

    pub fn multiply(&self, other: &Tensor) -> Tensor {
        let (a_val, b_val) = (
            self.data.borrow().value,
            other.data.borrow().value
            );
        let result = Tensor::new(a_val * b_val);

        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Multiply;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }

    pub fn sigmoid(&self) -> Tensor {
        let val = self.data.borrow().value;
        let result = Tensor::new(1.0 / (1.0 + (-val).exp()));

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
            data.grad = 1.0;
        }
        self._backward();
    }

    fn _backward(&self){
        let data = self.data.borrow();
        match data.operation{
            Operation::Add => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                a.data.borrow_mut().grad += data.grad;
                b.data.borrow_mut().grad += data.grad;
            },

            Operation::Sub => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                a.data.borrow_mut().grad += data.grad;
                b.data.borrow_mut().grad -= data.grad;
            }

            Operation::Multiply => {
                let a = &data.dependencies[0];
                let b = &data.dependencies[1];
                let a_val = a.data.borrow().value;
                let b_val = b.data.borrow().value;
                a.data.borrow_mut().grad += b_val * data.grad;
                b.data.borrow_mut().grad += a_val * data.grad;
            },

            Operation::Sigmoid => {
                let x = &data.dependencies[0];
                let s = data.value;
                x.data.borrow_mut().grad += s * (1.0 - s) * data.grad;
            }

            Operation::None => {}
        }

        for dep in &data.dependencies {
            dep._backward();
        }
    }
}