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

pub fn matrix_multiply(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = a[0].len();
    let p = b[0].len();

    assert_eq!(n, b.len(), "Matrix multiplication dimension mismatch");

    let mut result = vec![vec![0.0; p]; m];

    for i in 0..m{
        for k in 0..n {
            let a_ik = a[i][k];
            for j in 0..p {
                result[i][j] += a_ik * b[k][j];
            }
        }
    }
    result
}