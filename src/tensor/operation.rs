#[derive(Clone, Copy, Debug)]
pub enum Operation{
    None,
    Add,
    Sub,
    Multiply,
    Div,
    Matmul,
    Sigmoid,
    Mean,
    Pow(f64),
    ReLU,
    Log(f64),
    // TODO
}