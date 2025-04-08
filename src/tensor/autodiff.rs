use crate::tensor::value::TensorValue;
use super::{Tensor, NodeData, ops};

impl Tensor {
    pub fn backward(&self) {
        {
            let mut data = self.data.borrow_mut();
            data.grad = match &data.value {
                TensorValue::Scalar(_) => TensorValue::Scalar(1.0),
                TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![1.0; v.len()]),
                TensorValue::Matrix2D(m ) => {
                    TensorValue::Matrix2D(vec![vec![1.0; m[0].len()]; m.len()])
                }
                TensorValue::Tensor3D(t) => todo!()
            };
        }

        let mut nodes = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self._build_topo(&mut nodes, &mut visited);
        for node in nodes.iter().rev() {
            ops::_backward(node);
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
}