use crate::gluing::{GluingChecker, GluingError};
use crate::grammar::{DPOSpan, Graph, NodeId};
use std::collections::HashMap;

pub struct PushoutEngine;

impl PushoutEngine {
    pub fn apply_rewrite(span: &DPOSpan, g: &Graph, match_m: &HashMap<NodeId, NodeId>) -> Result<Graph, GluingError> {
        GluingChecker::check(&span.l, &span.k, g, match_m)?;

        let mut d = g.clone();
        let deleted_l_nodes: Vec<NodeId> = span.l.nodes.difference(&span.k.nodes).copied().collect();
        for l_node in deleted_l_nodes {
            if let Some(&g_node) = match_m.get(&l_node) {
                d.nodes.remove(&g_node);
                d.node_labels.remove(&g_node);
                d.edges.retain(|&(u, v)| u != g_node && v != g_node);
            }
        }

        let mut h = d;
        let mut r_to_h_map: HashMap<NodeId, NodeId> = HashMap::new();

        for &k_node in &span.k.nodes {
            if let Some(&g_node) = match_m.get(&k_node) {
                r_to_h_map.insert(k_node, g_node);
            }
        }

        let mut next_new_id = (h.nodes.iter().max().copied().unwrap_or(0)) + 1000;
        let added_r_nodes: Vec<NodeId> = span.r.nodes.difference(&span.k.nodes).copied().collect();

        for r_node in added_r_nodes {
            let new_id = next_new_id;
            next_new_id += 1;
            r_to_h_map.insert(r_node, new_id);
            let label = span.r.node_labels.get(&r_node).copied().unwrap_or(0);
            h.add_node(new_id, label);
        }

        // Safe lookups: return a typed GluingError instead of panicking
        for &(u, v) in &span.r.edges {
            let h_u = *r_to_h_map.get(&u).ok_or(GluingError::MalformedRule { u, v })?;
            let h_v = *r_to_h_map.get(&v).ok_or(GluingError::MalformedRule { u, v })?;
            h.add_edge(h_u, h_v);
        }

        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_deletes_node_and_adds_node() {
        // L: a -> b ; K: a ; R: a -> c
        let mut l = Graph::new();
        l.add_node(0, 0);
        l.add_node(1, 0);
        l.add_edge(0, 1);
        let mut k = Graph::new();
        k.add_node(0, 0);
        let mut r = Graph::new();
        r.add_node(0, 0);
        r.add_node(2, 7);
        r.add_edge(0, 2);
        let span = DPOSpan { l, k, r };

        let mut g = Graph::new();
        g.add_node(10, 0);
        g.add_node(11, 0);
        g.add_edge(10, 11);
        let m: HashMap<NodeId, NodeId> = [(0, 10), (1, 11)].into_iter().collect();

        let h = PushoutEngine::apply_rewrite(&span, &g, &m).unwrap();
        assert!(!h.nodes.contains(&11));
        assert!(h.nodes.contains(&10));
        assert_eq!(h.nodes.len(), 2);
        assert_eq!(h.edges.len(), 1);
    }

    #[test]
    fn test_malformed_rule_returns_error_instead_of_panic() {
        // K contains node 0, but the match does not map it -> R's edge (0, 2) cannot be resolved
        let mut l = Graph::new();
        l.add_node(0, 0);
        let mut k = Graph::new();
        k.add_node(0, 0);
        let mut r = Graph::new();
        r.add_node(0, 0);
        r.add_node(2, 0);
        r.add_edge(0, 2);
        let span = DPOSpan { l, k, r };

        let mut g = Graph::new();
        g.add_node(10, 0);
        let m: HashMap<NodeId, NodeId> = HashMap::new();

        assert_eq!(
            PushoutEngine::apply_rewrite(&span, &g, &m),
            Err(GluingError::MalformedRule { u: 0, v: 2 })
        );
    }
}
