use crate::grammar::Graph;
use std::collections::{HashSet, VecDeque};

pub struct ReachabilityChecker;

impl ReachabilityChecker {
    pub fn is_reachable(graph: &Graph, start: usize, target: usize) -> bool {
        if start == target {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(curr) = queue.pop_front() {
            if curr == target {
                return true;
            }
            for &(u, v) in &graph.edges {
                if u == curr && !visited.contains(&v) {
                    visited.insert(v);
                    queue.push_back(v);
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reachability() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        assert!(ReachabilityChecker::is_reachable(&g, 1, 3));
        assert!(!ReachabilityChecker::is_reachable(&g, 3, 1));
    }
}
