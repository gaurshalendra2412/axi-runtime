use std::collections::{HashMap, HashSet};

pub type NodeId = usize;
pub type EdgeId = (NodeId, NodeId);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Graph {
    pub nodes: HashSet<NodeId>,
    pub edges: HashSet<EdgeId>,
    pub node_labels: HashMap<NodeId, i32>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, id: NodeId, label: i32) {
        self.nodes.insert(id);
        self.node_labels.insert(id, label);
    }

    pub fn add_edge(&mut self, u: NodeId, v: NodeId) {
        self.nodes.insert(u);
        self.nodes.insert(v);
        self.edges.insert((u, v));
    }
}

#[derive(Debug, Clone)]
pub struct DPOSpan {
    pub l: Graph,
    pub k: Graph,
    pub r: Graph,
}
