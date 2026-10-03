use std::collections::HashMap;

pub struct RandomWalkPositionalEncoding;

impl RandomWalkPositionalEncoding {
    pub fn compute(
        adj: &HashMap<(usize, usize), Vec<(usize, usize)>>,
        nodes: &[(usize, usize)],
        k_steps: usize,
    ) -> HashMap<(usize, usize), Vec<f64>> {
        let mut rwpe = HashMap::new();
        for &node in nodes {
            let mut step_probs = vec![0.0; k_steps];
            let mut curr_dist: HashMap<(usize, usize), f64> = HashMap::new();
            curr_dist.insert(node, 1.0);

            for step in 0..k_steps {
                let mut next_dist: HashMap<(usize, usize), f64> = HashMap::new();
                for (u, prob) in &curr_dist {
                    if let Some(neighbors) = adj.get(u) {
                        if !neighbors.is_empty() {
                            let transition_prob = prob / (neighbors.len() as f64);
                            for &v in neighbors {
                                *next_dist.entry(v).or_insert(0.0) += transition_prob;
                            }
                        }
                    }
                }
                step_probs[step] = *next_dist.get(&node).unwrap_or(&0.0);
                curr_dist = next_dist;
            }
            rwpe.insert(node, step_probs);
        }
        rwpe
    }
}
