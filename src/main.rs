use num::complex::Complex;

mod matrix;

fn main() {
    let m1: matrix::Matrix<Complex<f64>> = matrix::Matrix::new(
        vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(-1.0, 0.0),
            Complex::new(0.0, -1.0),
        ],
        [2, 2],
    )
    .expect("fail");
    println!("{}", m1);
}
