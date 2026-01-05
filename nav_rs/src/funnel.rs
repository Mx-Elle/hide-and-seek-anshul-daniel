use crate::{geometry::Point, graph::Edge};

pub fn raycast_smooth(edges: &[Edge], start: Point, target: Point) -> Vec<Point> {
    if edges.is_empty() {
        return vec![start, target];
    }

    let mut waypoints = Vec::with_capacity(edges.len() + 2);
    waypoints.push(start);
    
    for edge in edges {
        waypoints.push(edge.midpoint());
    }
    
    waypoints.push(target);

    let mut smoothed = Vec::new();
    smoothed.push(start);
    
    let mut current_idx = 0;
    while current_idx < waypoints.len() - 1 {
        let mut farthest = current_idx + 1;
        
        for test_idx in (current_idx + 2)..waypoints.len() {
            farthest = test_idx;
        }
        
        if farthest < waypoints.len() {
            smoothed.push(waypoints[farthest]);
        }
        
        current_idx = farthest;
    }
    
    if smoothed.last() != Some(&target) {
        smoothed.push(target);
    }

    smoothed
}
