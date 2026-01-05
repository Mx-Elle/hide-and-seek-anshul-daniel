use crate::geometry::{self, Point};
use numpy::{Element, PyArrayDescr, dtype};
use pyo3::prelude::*;

#[repr(C)]
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Node {
    #[pyo3(get, set)]
    pub centroid: Point,
    #[pyo3(get, set)]
    pub edge_start: usize,
    #[pyo3(get, set)]
    pub edge_count: usize,
}

#[pymethods]
impl Node {
    #[new]
    fn new(centroid: Point, edge_start: usize, edge_count: usize) -> Self {
        Node {
            centroid,
            edge_start,
            edge_count,
        }
    }
}

impl Node {
    #[inline(always)]
    pub fn find_edge_to<'a>(&self, edges: &'a [Edge], target_node_idx: usize) -> Option<&'a Edge> {
        let slice = &edges[self.edge_start..self.edge_start + self.edge_count];
        slice.iter().find(|e| e.to == target_node_idx)
    }
}

unsafe impl Element for Node {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::new(
            py,
            [
                ("centroid", dtype::<Point>(py)),
                ("edge_start", dtype::<usize>(py)),
                ("edge_count", dtype::<usize>(py)),
            ],
        )
        .unwrap()
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }
}

#[repr(C)]
#[pyclass]
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    #[pyo3(get, set)]
    pub to: usize,
    #[pyo3(get, set)]
    pub cost: u32,
    #[pyo3(get, set)]
    pub left: Point,
    #[pyo3(get, set)]
    pub right: Point,
}

#[pymethods]
impl Edge {
    #[new]
    fn new(to: usize, cost: u32, left: Point, right: Point) -> Self {
        Edge {
            to,
            cost,
            left,
            right,
        }
    }
}

impl Edge {
    #[inline(always)]
    pub fn midpoint(&self) -> Point {
        geometry::midpoint(self.left, self.right)
    }
}

unsafe impl Element for Edge {
    const IS_COPY: bool = true;

    fn get_dtype(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        PyArrayDescr::new(
            py,
            [
                ("to", dtype::<usize>(py)),
                ("cost", dtype::<u32>(py)),
                ("left", dtype::<Point>(py)),
                ("right", dtype::<Point>(py)),
                ("padding", dtype::<u32>(py)),
            ],
        )
        .unwrap()
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }
}
