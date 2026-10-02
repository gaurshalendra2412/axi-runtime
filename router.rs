use crate::tuple::{CoordinateTuple, EdgeTuple};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepresentationFormat {
    SparseTuples,
    PackedBitMatrix,
}

pub struct DensityRouter {
    pub threshold: f64,
}

impl Default for DensityRouter {
    fn default() -> Self {
        Self { threshold: 0.65 }
    }
}

impl DensityRouter {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn compute_density(nodes: &[CoordinateTuple], edges: &[EdgeTuple], shape: (usize, usize)) -> f64 {
        let (rows, cols) = shape;
        let max_entities = (rows * cols) as f64;
        if max_entities == 0.0 {
            return 0.0;
        }
        (nodes.len() + edges.len()) as f64 / max_entities
    }

    pub fn route(&self, nodes: &[CoordinateTuple], edges: &[EdgeTuple], shape: (usize, usize)) -> RepresentationFormat {
        if Self::compute_density(nodes, edges, shape) <= self.threshold {
            RepresentationFormat::SparseTuples
        } else {
            RepresentationFormat::PackedBitMatrix
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_routing_threshold() {
        let router = DensityRouter::new(0.65);
        let nodes = vec![CoordinateTuple::new(0, 0, 1)];
        let edges: Vec<EdgeTuple> = Vec::new();
        // 1 active entity on a 10x10 grid = 0.01 density <= 0.65 -> Sparse
        assert_eq!(router.route(&nodes, &edges, (10, 10)), RepresentationFormat::SparseTuples);
    }

    #[test]
    fn test_dense_routes_to_packed() {
        let router = DensityRouter::new(0.65);
        let nodes: Vec<CoordinateTuple> = (0..4).map(|i| CoordinateTuple::new(i / 2, i % 2, 1)).collect();
        let edges: Vec<EdgeTuple> = Vec::new();
        // 4 entities on a 2x2 grid = 1.0 density > 0.65 -> Packed
        assert_eq!(router.route(&nodes, &edges, (2, 2)), RepresentationFormat::PackedBitMatrix);
    }
}
