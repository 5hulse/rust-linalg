use num::Num;
use std::fmt;
use std::ops::{Add, Mul, Sub};

#[derive(Debug)]
pub struct Matrix<T: Num + Copy> {
    data: Vec<T>,
    shape: [usize; 2],
    strides: [usize; 2],
}

impl<T: Num + Copy> Matrix<T> {
    pub fn new(data: Vec<T>, shape: [usize; 2]) -> Result<Self, String> {
        if data.len() != shape.iter().product() {
            return Err(format!(
                "Incompatible shape ({}) for data of size {}.",
                shape_str(&shape),
                data.len(),
            ));
        }

        Ok(Self {
            data,
            shape,
            strides: [shape[1], 1],
        })
    }

    pub fn new_vector(data: Vec<T>) -> Self {
        let length = data.len();
        Matrix::<T> {
            data,
            shape: [length, 1],
            strides: [1, 1],
        }
    }

    pub fn zeros(shape: [usize; 2]) -> Self {
        let data = vec![T::zero(); shape.iter().product()];
        Self {
            data,
            shape,
            strides: [shape[1], 1],
        }
    }

    pub fn ones(shape: [usize; 2]) -> Self {
        let data = vec![T::one(); shape.iter().product()];
        Self {
            data,
            shape,
            strides: [shape[1], 1],
        }
    }

    pub fn uniform(value: T, shape: [usize; 2]) -> Self {
        let data = vec![value * T::one(); shape.iter().product()];
        Self {
            data,
            shape,
            strides: [shape[1], 1],
        }
    }

    pub fn eye(size: usize) -> Self {
        let mut data = vec![T::zero(); size * size];
        let mut idx: usize;

        for i in 0..size {
            idx = i * (size + 1);
            data[idx] = T::one();
        }

        Self {
            data,
            shape: [size, size],
            strides: [size, 1],
        }
    }

    pub fn transpose(&mut self) {
        (self.shape[0], self.shape[1]) = (self.shape[1], self.shape[0]);
        (self.strides[0], self.strides[1]) = (self.strides[1], self.strides[0]);
    }

    fn row_indices(&self) -> Vec<Vec<usize>> {
        let mut rows: Vec<Vec<usize>> = Vec::with_capacity(self.shape[0]);
        let mut start: usize;
        let mut jump: usize;
        for r in 0..self.shape[0] {
            let mut row: Vec<usize> = Vec::with_capacity(self.shape[1]);
            start = r * self.strides[0];
            for c in 0..self.shape[1] {
                jump = c * self.strides[1];
                row.push(start + jump);
            }
            rows.push(row);
        }

        rows
    }

    pub fn rows(&self) -> Vec<Vec<T>> {
        let indicess = self.row_indices();
        let mut rows: Vec<Vec<T>> = Vec::with_capacity(self.shape[0]);
        for indices in indicess {
            let mut row: Vec<T> = Vec::with_capacity(self.shape[1]);
            for idx in indices {
                row.push(self.data[idx]);
            }
            rows.push(row);
        }

        rows
    }

    fn col_indices(&self) -> Vec<Vec<usize>> {
        let mut cols: Vec<Vec<usize>> = Vec::with_capacity(self.shape[1]);
        let mut start: usize;
        let mut jump: usize;
        for c in 0..self.shape[1] {
            let mut col: Vec<usize> = Vec::with_capacity(self.shape[0]);
            start = c * self.strides[1];
            for r in 0..self.shape[0] {
                jump = r * self.strides[0];
                col.push(start + jump);
            }
            cols.push(col);
        }

        cols
    }

    pub fn cols(&self) -> Vec<Vec<T>> {
        let indicess = self.col_indices();
        let mut cols: Vec<Vec<T>> = Vec::with_capacity(self.shape[1]);
        for indices in indicess {
            let mut col: Vec<T> = Vec::with_capacity(self.shape[0]);
            for idx in indices {
                col.push(self.data[idx]);
            }
            cols.push(col);
        }

        cols
    }

    pub fn ref_add(&self, other: &Self) -> Result<Matrix<T>, String> {
        self.check_identical_sizes(&other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.size());
        for (row_a, row_b) in self.rows().into_iter().zip(other.rows().into_iter()) {
            for (a, b) in row_a.into_iter().zip(row_b.into_iter()) {
                data.push(a + b);
            }
        }

        Ok(Matrix {
            data,
            shape: self.shape.clone(),
            strides: [self.shape[1], 1],
        })
    }

    pub fn into_add(self, other: Self) -> Result<Matrix<T>, String> {
        self.check_identical_sizes(&other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.size());
        for (row_a, row_b) in self.rows().into_iter().zip(other.rows().into_iter()) {
            for (a, b) in row_a.into_iter().zip(row_b.into_iter()) {
                data.push(a + b);
            }
        }

        Ok(Matrix {
            data,
            shape: self.shape.clone(),
            strides: [self.shape[1], 1],
        })
    }

    pub fn ref_sub(&self, other: &Self) -> Result<Matrix<T>, String> {
        self.check_identical_sizes(other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.size());
        for (row_a, row_b) in self.rows().into_iter().zip(other.rows().into_iter()) {
            for (a, b) in row_a.into_iter().zip(row_b.into_iter()) {
                data.push(a - b);
            }
        }

        Ok(Matrix {
            data,
            shape: self.shape.clone(),
            strides: [self.shape[1], 1],
        })
    }

    pub fn into_sub(self, other: Self) -> Result<Matrix<T>, String> {
        self.check_identical_sizes(&other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.size());
        for (row_a, row_b) in self.rows().into_iter().zip(other.rows().into_iter()) {
            for (a, b) in row_a.into_iter().zip(row_b.into_iter()) {
                data.push(a - b);
            }
        }

        Ok(Matrix {
            data,
            shape: self.shape,
            strides: [self.shape[1], 1],
        })
    }

    pub fn ref_mul(&self, other: &Self) -> Result<Matrix<T>, String> {
        self.check_muliplication_pair(other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.shape[0] * other.shape[1]);
        let mut value: T;
        for row_a in self.rows().iter() {
            for col_b in other.cols().iter() {
                value = T::zero();
                for (a, b) in row_a.iter().zip(col_b.iter()) {
                    value = value + (*a * *b);
                }
                data.push(value);
            }
        }

        Ok(Matrix {
            data,
            shape: [self.shape[0], other.shape[1]],
            strides: [other.shape[1], 1],
        })
    }

    pub fn into_mul(self, other: Self) -> Result<Matrix<T>, String> {
        self.check_muliplication_pair(&other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.shape[0] * other.shape[1]);
        let mut value: T;
        for row_a in self.rows().iter() {
            for col_b in other.cols().iter() {
                value = T::zero();
                for (a, b) in row_a.iter().zip(col_b.iter()) {
                    value = value + (*a * *b);
                }
                data.push(value);
            }
        }

        Ok(Matrix {
            data,
            shape: [self.shape[0], other.shape[1]],
            strides: [other.shape[1], 1],
        })
    }

    fn size(&self) -> usize {
        self.shape.iter().product()
    }

    fn check_identical_sizes(&self, other: &Self) -> Result<(), String> {
        match self.shape == other.shape {
            true => Ok(()),
            false => Err(format!(
                "Incompatible shapes: ({}) vs ({}).",
                shape_str(&self.shape),
                shape_str(&other.shape),
            )),
        }
    }

    fn check_muliplication_pair(&self, other: &Self) -> Result<(), String> {
        match self.shape[1] == other.shape[0] {
            true => Ok(()),
            false => Err(format!(
                "Incompatible shapes: ({}) vs ({}).",
                shape_str(&self.shape),
                shape_str(&other.shape),
            )),
        }
    }
}

impl<T: Num + Copy + fmt::Display> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s: String = String::from("[");
        for row in self.rows() {
            s.push('[');
            s.push_str(
                &row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
            s.push_str("]\n");
        }
        s.pop();
        s.push(']');
        write!(f, "{}", s)
    }
}

impl<T: Num + Copy> Add for Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: Self) -> Self::Output {
        match self.into_add(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T: Num + Copy> Add for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, other: Self) -> Self::Output {
        match self.ref_add(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T: Num + Copy> Mul for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: Self) -> Self::Output {
        match self.into_mul(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T: Num + Copy> Mul for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, other: Self) -> Self::Output {
        match self.ref_mul(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T: Num + Copy> Sub for Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: Self) -> Self::Output {
        match self.into_sub(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T: Num + Copy> Sub for &Matrix<T> {
    type Output = Matrix<T>;

    fn sub(self, other: Self) -> Self::Output {
        match self.ref_sub(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T: Num + Copy> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.data == other.data) && (self.shape == other.shape) && (self.strides == other.strides)
    }

    fn ne(&self, other: &Self) -> bool {
        (self.shape != other.shape) || (self.data != other.data) || (self.strides != other.strides)
    }
}

fn shape_str(shape: &[usize; 2]) -> String {
    shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

#[cfg(test)]
pub mod tests {
    use crate::matrix::*;
    use num::Complex;

    fn m_2by2_f64_1() -> Matrix<f64> {
        Matrix::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap()
    }

    fn m_2by2_f64_2() -> Matrix<f64> {
        Matrix::new(vec![-1.0, -2.0, -3.0, -4.0], [2, 2]).unwrap()
    }

    fn v_2by1_f64() -> Matrix<f64> {
        Matrix::new(vec![-1.0, -2.0], [2, 1]).unwrap()
    }

    #[test]
    pub fn test_new() {
        assert_eq!(
            Matrix::<f64>::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]).expect("fail"),
            Matrix {
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: [2, 2],
                strides: [2, 1],
            }
        );
        assert_eq!(
            Matrix::<Complex<f64>>::new(
                vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 1.0),
                    Complex::new(-1.0, 1.0),
                    Complex::new(0.0, 0.0),
                ],
                [2, 2]
            )
            .expect("new fail"),
            Matrix::<Complex<f64>> {
                data: vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 1.0),
                    Complex::new(-1.0, 1.0),
                    Complex::new(0.0, 0.0),
                ],
                shape: [2, 2],
                strides: [2, 1],
            }
        );

        assert_eq!(
            Matrix::<f64>::new(vec![1.0, 0.0, 1.0], [2, 1]).unwrap_err(),
            "Incompatible shape (2x1) for data of size 3.",
        );
    }

    #[test]
    pub fn test_new_vector() {
        assert_eq!(
            Matrix::<i32>::new_vector(vec![1, 2, 3, 4]),
            Matrix::<i32> {
                data: vec![1, 2, 3, 4],
                shape: [4, 1],
                strides: [1, 1],
            }
        );
    }

    #[test]
    pub fn test_zeros() {
        assert_eq!(
            Matrix::<f64>::zeros([2, 3]),
            Matrix::<f64> {
                data: [0.0; 6].to_vec(),
                shape: [2, 3],
                strides: [3, 1],
            }
        );
    }

    #[test]
    pub fn test_ones() {
        assert_eq!(
            Matrix::<Complex<f64>>::ones([5, 1]),
            Matrix::<Complex<f64>> {
                data: [Complex::new(1.0, 0.0); 5].to_vec(),
                shape: [5, 1],
                strides: [1, 1],
            }
        );
    }

    #[test]
    pub fn test_uniform() {
        assert_eq!(
            Matrix::<Complex<f64>>::uniform(Complex::new(std::f64::consts::PI, 0.0), [4, 2]),
            Matrix::<Complex<f64>> {
                data: [Complex::new(std::f64::consts::PI, 0.0); 8].to_vec(),
                shape: [4, 2],
                strides: [2, 1],
            }
        );
    }

    #[test]
    pub fn test_eye() {
        assert_eq!(
            Matrix::<Complex<f64>>::eye(3),
            Matrix {
                data: vec![
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(1.0, 0.0),
                ],
                shape: [3, 3],
                strides: [3, 1],
            }
        );
    }

    #[test]
    pub fn test_rows_and_cols() {
        let mut m1: Matrix<f64> =
            Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]).expect("new fail");
        assert_eq!(m1.rows(), vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        assert_eq!(
            m1.cols(),
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        m1.transpose();
        assert_eq!(
            m1.rows(),
            vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]
        );
        assert_eq!(m1.cols(), vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    pub fn test_ref_add() {
        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        let m3 = Matrix::<f64>::new(vec![0.0, 0.0, 0.0, 0.0], [2, 2]).expect("new fail");
        let m4 = Matrix::<f64>::new(vec![0.0, -1.0, 1.0, 0.0], [2, 2]).expect("new fail");
        assert_eq!(m1.ref_add(&m2).expect("ref_add fail"), m3);
        m2.transpose();
        assert_eq!(m1.ref_add(&m2).expect("ref_add fail"), m4);

        let m5 = Matrix::<f64>::new(vec![0.0; 9], [3, 3]).expect("new fail");
        assert_eq!(
            m1.ref_add(&m5).unwrap_err(),
            "Incompatible shapes: (2x2) vs (3x3).",
        );

        m2.transpose();
        assert_eq!(&m1 + &m2, m3);
        m2.transpose();
        assert_eq!(&m1 + &m2, m4);
        let result = std::panic::catch_unwind(|| &m1 + &m5);
        assert!(result.is_err());
    }

    #[test]
    pub fn test_into_add() {
        let m1 = m_2by2_f64_1();
        let m2 = m_2by2_f64_2();
        let m3 = Matrix::new(vec![0.0, 0.0, 0.0, 0.0], [2, 2]).expect("new fail");
        assert_eq!(m1.into_add(m2).expect("into_add fail"), m3);

        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        m2.transpose();
        let m4 = Matrix::<f64>::new(vec![0.0, -1.0, 1.0, 0.0], [2, 2]).expect("new fail");
        assert_eq!(m1.into_add(m2).expect("ref_add fail"), m4);

        let m1 = m_2by2_f64_1();
        let m5 = Matrix::<f64>::new(vec![0.0; 9], [3, 3]).expect("new fail");
        assert_eq!(
            m1.into_add(m5).unwrap_err(),
            "Incompatible shapes: (2x2) vs (3x3).",
        );

        let m1 = m_2by2_f64_1();
        let m2 = m_2by2_f64_2();
        assert_eq!(m1 + m2, m3);

        let m1 = m_2by2_f64_1();
        let m5 = Matrix::<f64>::new(vec![0.0; 9], [3, 3]).expect("new fail");
        let result = std::panic::catch_unwind(|| m1 + m5);
        assert!(result.is_err());
    }

    #[test]
    pub fn test_ref_sub() {
        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        let m3 = Matrix::<f64>::new(vec![2.0, 4.0, 6.0, 8.0], [2, 2]).expect("new fail");
        assert_eq!(&m1.ref_sub(&m2).expect("ref_sub fail"), &m3);

        m2.transpose();
        let m4 = Matrix::<f64>::new(vec![2.0, 5.0, 5.0, 8.0], [2, 2]).expect("new fail");
        assert_eq!(&m1.ref_sub(&m2).expect("ref_sub fail"), &m4);

        let m5 = Matrix::<f64>::new(vec![0.0; 6], [3, 2]).expect("new fail");
        assert_eq!(
            &m1.ref_add(&m5).unwrap_err(),
            "Incompatible shapes: (2x2) vs (3x2).",
        );

        m2.transpose();
        assert_eq!(&m1 - &m2, m3);

        m2.transpose();
        assert_eq!(&m1 - &m2, m4);

        let result = std::panic::catch_unwind(|| &m1 - &m5);
        assert!(result.is_err());
    }

    #[test]
    pub fn test_into_sub() {
        let m1 = m_2by2_f64_1();
        let m2 = m_2by2_f64_2();
        let m3 = Matrix::<f64>::new(vec![2.0, 4.0, 6.0, 8.0], [2, 2]).expect("new fail");
        assert_eq!(m1.into_sub(m2).expect("into_sub fail"), m3);

        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        m2.transpose();
        let m4 = Matrix::<f64>::new(vec![2.0, 5.0, 5.0, 8.0], [2, 2]).expect("new fail");
        assert_eq!(m1.into_sub(m2).expect("into_sub fail"), m4);

        let m1 = m_2by2_f64_1();
        let m5 = Matrix::<f64>::new(vec![0.0; 6], [3, 2]).expect("new fail");
        assert_eq!(
            m1.into_add(m5).unwrap_err(),
            "Incompatible shapes: (2x2) vs (3x2).",
        );

        let m1 = m_2by2_f64_1();
        let m2 = m_2by2_f64_2();
        let m3 = Matrix::<f64>::new(vec![2.0, 4.0, 6.0, 8.0], [2, 2]).expect("new fail");
        assert_eq!(m1 - m2, m3);

        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        m2.transpose();
        let m4 = Matrix::<f64>::new(vec![2.0, 5.0, 5.0, 8.0], [2, 2]).expect("new fail");
        assert_eq!(m1 - m2, m4);

        let m1 = m_2by2_f64_1();
        let m5 = Matrix::<f64>::new(vec![0.0; 6], [3, 2]).expect("new fail");
        let result = std::panic::catch_unwind(|| m1 - m5);
        assert!(result.is_err());
    }

    #[test]
    pub fn test_ref_mul() {
        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        let m3 = Matrix::<f64>::new(vec![-7.0, -10.0, -15.0, -22.0], [2, 2]).expect("new fail");
        assert_eq!(m1.ref_mul(&m2).expect("ref_mul fail"), m3);

        m2.transpose();
        let m4 = Matrix::<f64>::new(vec![-5.0, -11.0, -11.0, -25.0], [2, 2]).expect("new fail");
        assert_eq!(m1.ref_mul(&m2).expect("ref_mul fail"), m4);

        let v1 = v_2by1_f64();
        let m5 = Matrix::<f64>::new(vec![-5.0, -11.0], [2, 1]).expect("new fail");
        assert_eq!(m1.ref_mul(&v1).expect("ref_mul fail"), m5);

        let m6 =
            Matrix::<f64>::new(vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [3, 2]).expect("new fail");
        assert_eq!(
            m1.ref_mul(&m6).unwrap_err(),
            "Incompatible shapes: (2x2) vs (3x2)."
        );

        m2.transpose();
        assert_eq!(&m1 * &m2, m3);

        m2.transpose();
        assert_eq!(&m1 * &m2, m4);

        let result = std::panic::catch_unwind(|| &m1 * &m6);
        assert!(result.is_err());
    }

    #[test]
    pub fn test_into_mul() {
        let m1 = m_2by2_f64_1();
        let m2 = m_2by2_f64_2();
        let m3 = Matrix::<f64>::new(vec![-7.0, -10.0, -15.0, -22.0], [2, 2]).expect("new fail");
        assert_eq!(m1.into_mul(m2).expect("into_mul fail"), m3);

        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        m2.transpose();
        let m4 = Matrix::<f64>::new(vec![-5.0, -11.0, -11.0, -25.0], [2, 2]).expect("new fail");
        assert_eq!(m1.into_mul(m2).expect("into_mul fail"), m4);

        let m1 = m_2by2_f64_1();
        let v1 = v_2by1_f64();
        let m5 = Matrix::new(vec![-5.0, -11.0], [2, 1]).expect("new fail");
        assert_eq!(m1.into_mul(v1).expect("into_mul fail"), m5);

        let m1 = m_2by2_f64_1();
        let m5 = Matrix::<f64>::new(vec![0.0; 6], [3, 2]).expect("new fail");
        assert_eq!(
            m1.into_mul(m5).unwrap_err(),
            "Incompatible shapes: (2x2) vs (3x2).",
        );

        let m1 = m_2by2_f64_1();
        let m2 = m_2by2_f64_2();
        let m3 = Matrix::new(vec![-7.0, -10.0, -15.0, -22.0], [2, 2]).expect("new fail");
        assert_eq!(m1 * m2, m3);

        let m1 = m_2by2_f64_1();
        let mut m2 = m_2by2_f64_2();
        m2.transpose();
        let m4 = Matrix::new(vec![-5.0, -11.0, -11.0, -25.0], [2, 2]).expect("new fail");
        assert_eq!(m1 * m2, m4);

        let m1 = m_2by2_f64_1();
        let m5 = Matrix::new(vec![0.0; 6], [3, 2]).expect("new fail");
        let result = std::panic::catch_unwind(|| m1 * m5);
        assert!(result.is_err());
    }
}
