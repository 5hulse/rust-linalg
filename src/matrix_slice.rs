//matrix_slice.rs
//Simon Hulse
//simonhulse@protonmail.com
//Last Edited: Tue 11 Jun 2024 12:09:48 AM EDT

use crate::matrix::Matrix;
use approx::AbsDiffEq;
use num::Num;
use std::fmt;

pub struct SliceSpec {
    start: usize,
    stop: usize,
    jump: usize,
    gap: usize,
}

impl SliceSpec {
    pub fn new(start: usize, stop: usize, jump: usize) -> Self {
        let gap = usize::max(start, stop) - usize::min(start, stop);
        SliceSpec {
            start,
            stop,
            jump,
            gap,
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
    /// let spec2= SliceSpec::new(0, 12, 3);
    /// assert_eq!(spec2.get_indices(), vec![0, 3, 6, 9]);
    ///
    /// let spec3= SliceSpec::new(7, 1, 2);
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

        let capacity = (self.gap / self.jump) + 1;
        let mut indices = Vec::with_capacity(capacity);
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
