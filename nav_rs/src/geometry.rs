use numpy::{Element, PyArrayDescr, dtype};
use pyo3::prelude::*;

#[repr(C)]
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    #[pyo3(get, set)]
    pub x: f32,
    #[pyo3(get, set)]
    pub y: f32,
}

#[pymethods]
impl Point {
    #[new]
    fn new(x: f32, y: f32) -> Self {
        Point { x, y }
    }
}

unsafe impl Element for Point {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::new(py, [("x", dtype::<f32>(py)), ("y", dtype::<f32>(py))]).unwrap()
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }
}

#[inline(always)]
pub fn midpoint(p1: Point, p2: Point) -> Point {
    Point {
        x: (p1.x + p2.x) * 0.5,
        y: (p1.y + p2.y) * 0.5,
    }
}

#[inline(always)]
fn distance_squared(p1: Point, p2: Point) -> f32 {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    dx * dx + dy * dy
}

#[inline(always)]
pub fn distance(p1: Point, p2: Point) -> f32 {
    distance_squared(p1, p2).sqrt()
}
