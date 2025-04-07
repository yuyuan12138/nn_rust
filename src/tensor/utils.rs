pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result = vec![vec![0.0; rows]; cols];

    for i in 0..rows {
        for j in 0..cols{
            result[j][i] = matrix[i][j];
        }
    }
    result
}

pub fn matrix_multiply(a_mat: &Vec<Vec<f64>>, b_mat: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {

    let mut result = vec![vec![0.0; b_mat[0].len()]; a_mat.len()];

    for i in 0..a_mat.len() {
        for j in 0..b_mat[0].len() {
            for k in 0..a_mat[0].len() {
                result[i][j] += a_mat[i][k] * b_mat[k][j];
            }
        }
    }
    result
}