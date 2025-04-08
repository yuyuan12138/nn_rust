#[derive(Clone, Copy, Debug)]
pub enum Operation{
    None,

    Add,
    Sub,
    Multiply,
    Div,
    Matmul,
    Sigmoid,
    Tanh,
    Mean,
    Pow(f64),
    ReLU,
    Log(f64),
    Softmax,
    Sum,
    T,
    // TODO

    Broadcast,
    Unsqueeze(usize),
    Squeeze(Option<usize>),
}