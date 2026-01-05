use crate::astar::astar;
use crate::geometry::Point;
use crate::graph::{Edge, Node};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

mod astar;
mod funnel;
mod geometry;
mod graph;

#[pyfunction]
fn find_path(
    _py: Python<'_>,
    nodes: PyReadonlyArray1<Node>,
    edges: PyReadonlyArray1<Edge>,
    start: usize,
    target: usize,
    target_point: Point,
) -> PyResult<Vec<Point>> {
    let nodes_slice = nodes.as_slice().unwrap();
    let edges_slice = edges.as_slice().unwrap();

    let mut path_points = match astar(nodes_slice, edges_slice, start, target) {
        Some(points) => points,
        None => Vec::new(),
    };

    path_points.push(target_point);

    Ok(path_points)
}

#[pymodule]
fn nav_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Point>()?;
    m.add_class::<Node>()?;
    m.add_class::<Edge>()?;
    m.add_function(wrap_pyfunction!(find_path, m)?)?;
    Ok(())
}
