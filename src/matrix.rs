use approx::AbsDiffEq;
use num::Num;
use std::fmt;
use std::ops::{Add, Mul, Sub};

/// 2-dimensional matrix object.
#[derive(Debug)]
pub struct Matrix<T>
where
    T: Num + Copy + AbsDiffEq + fmt::Display,
{
    pub data: Vec<T>,
    pub shape: [usize; 2],
    pub strides: [usize; 2],
}

impl<T> Matrix<T>
where
    T: Num + Copy + AbsDiffEq + fmt::Display,
{
    /// Create a new matrix.
    ///
    /// # Arguments
    ///
    /// * `data` - A contiguous vector of the entries in the matrix, expressed in row-major order.
    /// * `shape` - The number of rows and columns in the matrix.
    ///
    /// # Returns
    ///
    /// `Ok(Matrix<T>)` if `data.len() == shape[0] * shape[1]`, and `Err(String)` otherwise.
    ///
    /// # Examples
    ///
    /// Creates a 90° rotation matrix:
    ///
    /// ```
    /// #[macro_use]
    /// extern crate approx;
    /// # fn main() {
    ///
    /// use std::f64::consts::FRAC_PI_2;
    /// use linalg::matrix::Matrix;
    ///
    /// let rot_90: Matrix<f64> = Matrix::new(
    ///     vec![
    ///         (FRAC_PI_2).cos(),
    ///         -(FRAC_PI_2).sin(),
    ///         (FRAC_PI_2).sin(),
    ///         (FRAC_PI_2).cos(),
    ///     ],
    ///     [2, 2],
    /// ).expect("Error calling `Matrix::new`");
    ///
    /// // `Matrix<T>` implements the `approx::AbsDiffEq` trait,
    /// // enabling comparison up to machine epsilon precision.
    /// assert_abs_diff_eq!(
    ///     rot_90,
    ///     Matrix::<f64> {
    ///         data: vec![0.0, -1.0, 1.0, 0.0],
    ///         shape: [2, 2],
    ///         strides: [2, 1],
    ///     }
    /// );
    /// # }
    /// ```
    ///
    /// Length of `data` doesn't comply with `shape`:
    ///
    /// ```
    /// use linalg::matrix::Matrix;
    ///
    /// let err_msg = Matrix::new(
    ///     vec![1.0, 2.0, 3.0],
    ///     [2, 2],
    /// ).unwrap_err();
    ///
    /// assert_eq!(err_msg, "Incompatible shape (2x2) for data of size 3.")
    /// ```
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

    /// Create a column (N x 1) vector.
    ///
    /// # Arguments
    ///
    /// * `data` - elements of the vector.
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let vector = Matrix::<f64>::new_vector(
    ///     vec![1.0, 2.0, 3.0],
    /// );
    ///
    /// assert_eq!(
    ///     vector,
    ///     Matrix {
    ///         data: vec![1.0, 2.0, 3.0],
    ///         shape: [3, 1],
    ///         strides: [1, 1],
    ///     },
    /// );
    pub fn new_vector(data: Vec<T>) -> Self {
        let length = data.len();
        Matrix::<T> {
            data,
            shape: [length, 1],
            strides: [1, 1],
        }
    }

    /// Create a matrix of zeros
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the matrix
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let zeros = Matrix::<f64>::zeros([3, 2]);
    ///
    /// assert_eq!(
    ///     zeros,
    ///     Matrix {
    ///         data: vec![0.0; 6],
    ///         shape: [3, 2],
    ///         strides: [2, 1],
    ///     },
    /// );
    pub fn zeros(shape: [usize; 2]) -> Self {
        let data = vec![T::zero(); shape.iter().product()];
        Self {
            data,
            shape,
            strides: [shape[1], 1],
        }
    }

    /// Create a matrix of ones
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the matrix
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let ones = Matrix::<i32>::ones([2, 3]);
    ///
    /// assert_eq!(
    ///     ones,
    ///     Matrix {
    ///         data: vec![1; 6],
    ///         shape: [2, 3],
    ///         strides: [3, 1],
    ///     },
    /// );
    pub fn ones(shape: [usize; 2]) -> Self {
        let data = vec![T::one(); shape.iter().product()];
        Self {
            data,
            shape,
            strides: [shape[1], 1],
        }
    }

    /// Create a matrix with identical elements
    ///
    /// # Arguments
    ///
    /// * `value` - Value for all elements to adopt
    /// * `shape` - Shape of the matrix
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// use std::f64::consts::PI;
    /// let pi_vector = Matrix::<f64>::uniform(PI, [4, 1]);
    ///
    /// assert_eq!(
    ///     pi_vector,
    ///     Matrix {
    ///         data: vec![PI; 4],
    ///         shape: [4, 1],
    ///         strides: [1, 1],
    ///     },
    /// );
    pub fn uniform(value: T, shape: [usize; 2]) -> Self {
        let data = vec![value * T::one(); shape.iter().product()];
        Self {
            data,
            shape,
            strides: [shape[1], 1],
        }
    }

    /// Create an identity matrix, i.e. a matrix with 1s along the main diagonal, and 0s elsewhere.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of rows (and columns).
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let i_2 = Matrix::<f64>::eye(2);
    ///
    /// assert_eq!(
    ///     i_2,
    ///     Matrix {
    ///         data: vec![1.0, 0.0, 0.0, 1.0],
    ///         shape: [2, 2],
    ///         strides: [2, 1],
    ///     },
    /// );
    ///
    /// let i_3 = Matrix::<f64>::eye(3);
    ///
    /// assert_eq!(
    ///     i_3,
    ///     Matrix {
    ///         data: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ///         shape: [3, 3],
    ///         strides: [3, 1],
    ///     },
    /// );
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

    /// Transpose a matrix in-place.
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let mut matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [2, 3],
    /// ).unwrap();
    ///
    /// matrix.transpose();
    ///
    /// assert_eq!(
    ///     matrix,
    ///     Matrix {
    ///         data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    ///         shape: [3, 2],
    ///         strides: [2, 1],
    ///     },
    /// );
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

    /// Return the rows that the matrix comprises
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let mut matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// assert_eq!(
    ///     matrix.rows(),
    ///     vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
    /// );
    ///
    /// matrix.transpose();
    ///
    /// assert_eq!(
    ///     matrix.rows(),
    ///     vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]],
    /// );
    /// ```
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

    /// Return the elements of the matrix as a contigous vector in row-major order.
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let mut matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [2, 3],
    /// ).unwrap();
    ///
    /// assert_eq!(
    ///     matrix.data_row_major(),
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// );
    ///
    /// matrix.transpose();
    ///
    /// assert_eq!(
    ///     matrix.data_row_major(),
    ///     vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    /// );
    /// ```
    pub fn data_row_major(&self) -> Vec<T> {
        let mut data = Vec::with_capacity(self.size());
        for row in self.rows().into_iter() {
            for elem in row.into_iter() {
                data.push(elem);
            }
        }

        data
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

    /// Return the columns that the matrix comprises.
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let mut matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// assert_eq!(
    ///     matrix.cols(),
    ///     vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]],
    /// );
    ///
    /// matrix.transpose();
    ///
    /// assert_eq!(
    ///     matrix.cols(),
    ///     vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
    /// );
    /// ```
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

    /// Return the elements of the matrix as a contigous vector in column-major order.
    ///
    /// # Example
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let mut matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [2, 3],
    /// ).unwrap();
    ///
    /// assert_eq!(
    ///     matrix.data_col_major(),
    ///     vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    /// );
    ///
    /// matrix.transpose();
    ///
    /// assert_eq!(
    ///     matrix.data_col_major(),
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    /// );
    /// ```
    pub fn data_col_major(&self) -> Vec<T> {
        let mut data = Vec::with_capacity(self.size());
        for col in self.cols().into_iter() {
            for elem in col.into_iter() {
                data.push(elem);
            }
        }

        data
    }

    /// Add two matrices.
    ///
    /// # Arguments
    ///
    /// * `other` - Reference to a `Matrix<T>` with the same `shape` as `self`.
    ///
    /// # Returns
    ///
    /// `Ok(Matrix<T>)` if `self.shape == other.shape`, otherwise `Err(String)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let m1 = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// let m2 = Matrix::<f64>::new(
    ///     vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// let m3 = m1.ref_add(&m2).unwrap();
    ///
    /// assert_eq!(
    ///     m3,
    ///     Matrix {
    ///         data: vec![0.0; 6],
    ///         shape: [3, 2],
    ///         strides: [2, 1],
    ///     }
    /// );
    /// ```
    ///
    /// Attempt to add two matrices of incompatible shapes:
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let m1 = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// let m2 = Matrix::<f64>::new(
    ///     vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    ///     [2, 3],
    /// ).unwrap();
    ///
    /// let err_msg = m1.ref_add(&m2).unwrap_err();
    ///
    /// assert_eq!(
    ///     err_msg,
    ///     "Incompatible shapes: (3x2) vs (2x3).",
    /// );
    /// ```
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

    /// Add two matrices; the matrices are moved into the method, and are no-longer available after
    /// it is called.
    ///
    /// # Arguments
    ///
    /// * `other` - `Matrix<T>` with the same `shape` as `self`.
    ///
    /// # Returns
    ///
    /// `Ok(Matrix<T>)` if `self.shape == other.shape`, otherwise `Err(String)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let m1 = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// let m2 = Matrix::<f64>::new(
    ///     vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// let m3 = m1.into_add(m2).unwrap();
    /// // m1 and m2 no longer available
    ///
    /// assert_eq!(
    ///     m3,
    ///     Matrix {
    ///         data: vec![0.0; 6],
    ///         shape: [3, 2],
    ///         strides: [2, 1],
    ///     }
    /// );
    /// ```
    ///
    /// Attempt to add two matrices of incompatible shapes:
    /// ```
    /// # use linalg::matrix::Matrix;
    /// let m1 = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     [3, 2],
    /// ).unwrap();
    ///
    /// let m2 = Matrix::<f64>::new(
    ///     vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
    ///     [2, 3],
    /// ).unwrap();
    ///
    /// let err_msg = m1.into_add(m2).unwrap_err();
    /// // m1 and m2 no longer available
    ///
    /// assert_eq!(
    ///     err_msg,
    ///     "Incompatible shapes: (3x2) vs (2x3).",
    /// );
    /// ```
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

    pub fn hadamard(&self, other: &Self) -> Result<Matrix<T>, String> {
        self.check_identical_sizes(&other)?;

        let mut data: Vec<T> = Vec::with_capacity(self.size());
        for (row_a, row_b) in self.rows().into_iter().zip(other.rows().into_iter()) {
            for (a, b) in row_a.into_iter().zip(row_b.into_iter()) {
                data.push(a * b);
            }
        }

        Ok(Matrix {
            data,
            shape: self.shape.clone(),
            strides: [self.shape[1], 1],
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

impl<T> fmt::Display for Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
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

impl<T> PartialEq for Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    fn eq(&self, other: &Self) -> bool {
        (self.data_row_major() == other.data_row_major()) && (self.shape == other.shape)
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl<T> Add for Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    type Output = Matrix<T>;

    fn add(self, other: Self) -> Self::Output {
        match self.into_add(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T> Add for &Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    type Output = Matrix<T>;

    fn add(self, other: Self) -> Self::Output {
        match self.ref_add(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T> Mul for Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    type Output = Matrix<T>;

    fn mul(self, other: Self) -> Self::Output {
        match self.into_mul(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T> Mul for &Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    type Output = Matrix<T>;

    fn mul(self, other: Self) -> Self::Output {
        match self.ref_mul(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    type Output = Matrix<T>;

    fn sub(self, other: Self) -> Self::Output {
        match self.into_sub(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T> Sub for &Matrix<T>
where
    T: Num + Copy + fmt::Display + AbsDiffEq,
{
    type Output = Matrix<T>;

    fn sub(self, other: Self) -> Self::Output {
        match self.ref_sub(other) {
            Ok(m) => m,
            Err(e) => panic!("{}", e),
        }
    }
}

impl<T> AbsDiffEq for Matrix<T>
where
    T: AbsDiffEq + Num + Copy + fmt::Display,
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        match self.shape == other.shape {
            true => (),
            false => return false,
        };
        self.data_row_major()
            .iter()
            .zip(other.data_row_major().iter())
            .all(|(a, b)| T::abs_diff_eq(a, b, epsilon))
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
            Matrix::<f64>::ones([5, 1]),
            Matrix::<f64> {
                data: vec![1.0; 5],
                shape: [5, 1],
                strides: [1, 1],
            }
        );
    }

    #[test]
    pub fn test_uniform() {
        assert_eq!(
            Matrix::<f64>::uniform(std::f64::consts::PI, [4, 2]),
            Matrix::<f64> {
                data: [std::f64::consts::PI; 8].to_vec(),
                shape: [4, 2],
                strides: [2, 1],
            }
        );
    }

    #[test]
    pub fn test_eye() {
        assert_eq!(
            Matrix::<f64>::eye(3),
            Matrix {
                data: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
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
