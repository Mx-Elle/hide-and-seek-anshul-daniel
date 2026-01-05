use crate::geometry::{Point, distance};
use crate::graph::{Edge, Node};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, Eq, PartialEq)]
struct State {
    f_score: u32,
    u: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_score
            .cmp(&self.f_score)
            .then_with(|| self.u.cmp(&other.u))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn astar(nodes: &[Node], edges: &[Edge], start: usize, target: usize) -> Option<Vec<Point>> {
    let nlen = nodes.len();
    let mut parents: Vec<Option<usize>> = vec![None; nlen];
    let mut g_score = vec![u32::MAX; nlen];
    let mut heap = BinaryHeap::new();

    g_score[start] = 0;
    heap.push(State {
        f_score: distance(nodes[start].centroid, nodes[target].centroid) as u32,
        u: start,
    });

    while let Some(State { f_score, u }) = heap.pop() {
        if u == target {
            return Some(reconstruct_path(nodes, edges, &parents, &start, &target));
        }

        let h_u = distance(nodes[u].centroid, nodes[target].centroid) as u32;
        if f_score > g_score[u] + h_u {
            continue;
        }

        let node = &nodes[u];
        let start = node.edge_start;
        let count = node.edge_count;

        for edge in &edges[start..start + count] {
            let v = edge.to;
            let g_v = g_score[u] + edge.cost;

            if g_score[v] <= g_v {
                continue;
            }

            g_score[v] = g_v;
            let h_v = distance(nodes[v].centroid, nodes[target].centroid) as u32;
            let f_v = g_v + h_v;
            parents[v] = Some(u);
            heap.push(State { f_score: f_v, u: v });
        }
    }

    None
}

fn reconstruct_path(
    nodes: &[Node],
    edges: &[Edge],
    parents: &[Option<usize>],
    start: &usize,
    target: &usize,
) -> Vec<Point> {
    let mut path = Vec::new();
    let mut current = *target;

    while let Some(parent) = parents[current] {
        let edge = nodes[parent].find_edge_to(edges, current);
        match edge {
            Some(edge) => {
                path.push(edge.midpoint());
            }
            None => break,
        }

        current = parent;
        if current == *start {
            break;
        }
    }

    path.reverse();
    path
}
