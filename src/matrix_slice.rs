//matrix_slice.rs
//Simon Hulse
//simonhulse@protonmail.com
//Last Edited: Tue 11 Jun 2024 15:48:46 PDT

use crate::matrix::Matrix;
use approx::AbsDiffEq;
use num::Num;
use std::fmt;

pub struct SliceSpec {
    start: usize,
    stop: usize,
    jump: usize,
    size: usize,
}

impl SliceSpec {
    pub fn new(start: usize, stop: usize, jump: usize) -> Self {
        let size = usize::max(start, stop) - usize::min(start, stop) / jump;
        SliceSpec {
            start,
            stop,
            jump,
            size,
        }
    }

    /// Retrieve the indices correspodning to the slice specification.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::linalg::matrix_slice::SliceSpec;
    /// let spec1 = SliceSpec::new(2, 9, 1);
    /// assert_eq!(spec1.get_indices(), vec![2, 3, 4, 5, 6, 7, 8]);
    ///
    /// let spec2 = SliceSpec::new(0, 12, 3);
    /// assert_eq!(spec2.get_indices(), vec![0, 3, 6, 9]);
    ///
    /// let spec3 = SliceSpec::new(7, 1, 2);
    /// assert_eq!(spec3.get_indices(), vec![7, 5, 3]);
    ///
    /// let spec4 = SliceSpec::new(0, 0, 1);
    /// assert_eq!(spec4.get_indices(), vec![]);
    /// ```
    pub fn get_indices(&self) -> Vec<usize> {
        // this feels weird...
        let op: Box<dyn Fn(usize) -> usize>;
        let check_break: Box<dyn Fn(usize) -> bool>;
        match self.start > self.stop {
            // Descending indices
            true => {
                op = Box::new(|curr| curr - self.jump);
                check_break = Box::new(|curr| curr <= self.stop);
            }
            // Ascending indices
            false => {
                op = Box::new(|curr| curr + self.jump);
                check_break = Box::new(|curr| curr >= self.stop);
            }
        };

        let mut indices = Vec::with_capacity(self.size);
        let mut current_value = self.start;

        loop {
            if check_break(current_value) {
                break;
            }
            indices.push(current_value);
            current_value = op(current_value);
        }

        indices
    }
}

pub struct MatrixSlice<'a, T>
where
    T: Num + Copy + AbsDiffEq + fmt::Display,
{
    matrix: &'a Matrix<T>,
    slice: [SliceSpec; 2],
}

impl<'a, T> MatrixSlice<'a, T>
where
    T: Num + Copy + AbsDiffEq + fmt::Display,
{
    // N.B. checks for a valid slice will be made within `Matrix`
    pub fn new(matrix: &'a Matrix<T>, slice: [SliceSpec; 2]) -> Self {
        Self { matrix, slice }
    }

    /// # Examples
    ///
    /// ```
    /// # use crate::linalg::matrix_slice::{MatrixSlice, SliceSpec};
    /// # use crate::linalg::matrix::Matrix;
    /// let matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     [3, 3],
    /// ).unwrap();
    ///
    /// let ms1 = MatrixSlice::new(
    ///     &matrix,
    ///     [SliceSpec::new(0, 2, 1), SliceSpec::new(1, 3, 1)],
    /// );
    ///
    /// assert_eq!(ms1.generate_indices(), vec![1, 2, 4, 5]);
    ///
    /// let ms2 = MatrixSlice::new(
    ///     &matrix,
    ///     [SliceSpec::new(2, 0, 1), SliceSpec::new(0, 3, 2)],
    /// );
    ///
    /// assert_eq!(ms2.generate_indices(), vec![6, 8, 3, 5]);
    /// ```
    pub fn generate_indices(&self) -> Vec<usize> {
        let d1_indices = self.slice[0].get_indices().into_iter();
        let d2_indices = self.slice[1].get_indices().into_iter();

        // Cartesian product of `d1_indices` and `d2_indices`
        let pairs = d1_indices
            .flat_map(move |d1| {
                d2_indices
                    .clone()
                    .map(move |d2| d1 * self.matrix.strides[0] + d2 * self.matrix.strides[1])
            })
            .collect();

        pairs
    }

    /// # Examples
    ///
    /// ```
    /// # use crate::linalg::matrix_slice::{MatrixSlice, SliceSpec};
    /// # use crate::linalg::matrix::Matrix;
    /// let matrix = Matrix::<f64>::new(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    ///     [3, 3],
    /// ).unwrap();
    ///
    /// let ms1 = MatrixSlice::new(
    ///     &matrix,
    ///     [SliceSpec::new(0, 2, 1), SliceSpec::new(1, 3, 1)],
    /// );
    ///
    /// assert_eq!(
    ///     ms1.yeild_matrix(),
    ///     Matrix::new(vec![2.0, 3.0, 5.0, 6.0], [2, 2]).unwrap()
    /// );
    /// ```
    pub fn yeild_matrix(&self) -> Matrix<T> {
        let data: Vec<T> = self
            .generate_indices()
            .into_iter()
            .map(|idx| self.matrix.data[idx])
            .collect();
        let shape = [self.slice[0].size, self.slice[1].size];

        Matrix::new(data, shape).unwrap()
    }
}
