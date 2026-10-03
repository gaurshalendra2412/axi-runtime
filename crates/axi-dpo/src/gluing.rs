use crate::grammar::{Graph, NodeId};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum GluingError {
    #[error("Dangling Edge Violation: Deleted node {deleted_node} remains attached to external edge ({u}, {v})")]
    DanglingEdge { deleted_node: NodeId, u: NodeId, v: NodeId },
    #[error("Identification Violation: Nodes {u} and {v} mapped to same target but not in K")]
    Identification { u: NodeId, v: NodeId },

    #[error("Malformed Rule Violation: Edge ({u}, {v}) in R references a node missing from interface mappings")]
    MalformedRule { u: NodeId, v: NodeId },
}

pub struct GluingChecker;

impl GluingChecker {
    pub fn check(l: &Graph, k: &Graph, g: &Graph, match_m: &HashMap<NodeId, NodeId>) -> Result<(), GluingError> {
        let mut reverse_match: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for (&l_node, &g_node) in match_m {
            reverse_match.entry(g_node).or_default().push(l_node);
        }

        for (&_g_node, l_nodes) in &reverse_match {
            if l_nodes.len() > 1 {
                for i in 0..l_nodes.len() {
                    for j in (i + 1)..l_nodes.len() {
                        let u = l_nodes[i];
                        let v = l_nodes[j];
                        if !k.nodes.contains(&u) || !k.nodes.contains(&v) {
                            return Err(GluingError::Identification { u, v });
                        }
                    }
                }
            }
        }

        let deleted_l_nodes: HashSet<NodeId> = l.nodes.difference(&k.nodes).copied().collect();
        let deleted_g_nodes: HashSet<NodeId> = deleted_l_nodes.iter().filter_map(|n| match_m.get(n).copied()).collect();

        for &(u, v) in &g.edges {
            let u_deleted = deleted_g_nodes.contains(&u);
            let v_deleted = deleted_g_nodes.contains(&v);

            if u_deleted || v_deleted {
                let is_matched_edge = l.edges.iter().any(|&(lu, lv)| {
                    match_m.get(&lu) == Some(&u) && match_m.get(&lv) == Some(&v)
                });

                if !is_matched_edge {
                    let deleted_node = if u_deleted { u } else { v };
                    return Err(GluingError::DanglingEdge { deleted_node, u, v });
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule_delete_b() -> (Graph, Graph) {
        // L: a -> b ; K: a (b is deleted)
        let mut l = Graph::new();
        l.add_node(0, 0);
        l.add_node(1, 0);
        l.add_edge(0, 1);
        let mut k = Graph::new();
        k.add_node(0, 0);
        (l, k)
    }

    #[test]
    fn test_dangling_edge_detected() {
        let (l, k) = rule_delete_b();
        let mut g = Graph::new();
        g.add_node(10, 0);
        g.add_node(11, 0);
        g.add_node(12, 0);
        g.add_edge(10, 11); // matched edge
        g.add_edge(12, 11); // external edge into the deleted node
        let m: HashMap<NodeId, NodeId> = [(0, 10), (1, 11)].into_iter().collect();
        assert!(matches!(
            GluingChecker::check(&l, &k, &g, &m),
            Err(GluingError::DanglingEdge { .. })
        ));
    }

    #[test]
    fn test_clean_deletion_passes() {
        let (l, k) = rule_delete_b();
        let mut g = Graph::new();
        g.add_node(10, 0);
        g.add_node(11, 0);
        g.add_edge(10, 11);
        let m: HashMap<NodeId, NodeId> = [(0, 10), (1, 11)].into_iter().collect();
        assert_eq!(GluingChecker::check(&l, &k, &g, &m), Ok(()));
    }
}
