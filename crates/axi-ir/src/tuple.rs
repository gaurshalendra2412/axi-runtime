use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CoordinateTuple {
    pub r: usize,
    pub c: usize,
    pub val: i32,
}

impl CoordinateTuple {
    pub fn new(r: usize, c: usize, val: i32) -> Self {
        Self { r, c, val }
    }
}

impl Ord for CoordinateTuple {
    fn cmp(&self, other: &Self) -> Ordering {
        self.r.cmp(&other.r).then_with(|| self.c.cmp(&other.c))
    }
}

impl PartialOrd for CoordinateTuple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EdgeTuple {
    pub u_r: usize,
    pub u_c: usize,
    pub v_r: usize,
    pub v_c: usize,
    pub relation: String,
    pub weight: i32,
}

impl EdgeTuple {
    pub fn new(u_r: usize, u_c: usize, v_r: usize, v_c: usize, relation: impl Into<String>, weight: i32) -> Self {
        Self { u_r, u_c, v_r, v_c, relation: relation.into(), weight }
    }
}
