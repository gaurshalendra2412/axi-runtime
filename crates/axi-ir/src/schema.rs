use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationalSchema {
    pub name: String,
    pub max_rows: usize,
    pub max_cols: usize,
    pub allowed_values: Vec<i32>,
}

impl RelationalSchema {
    pub fn validate_cell(&self, r: usize, c: usize, val: i32) -> bool {
        r < self.max_rows && c < self.max_cols && self.allowed_values.contains(&val)
    }
}
