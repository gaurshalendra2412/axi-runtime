use std::collections::HashMap;

pub type VirtualCoordId = (usize, usize);
pub type PhysicalPageId = usize;

#[derive(Default)]
pub struct TopologicalBlockTable {
    mapping: HashMap<VirtualCoordId, PhysicalPageId>,
}

impl TopologicalBlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn map(&mut self, virt_id: VirtualCoordId, phys_id: PhysicalPageId) {
        self.mapping.insert(virt_id, phys_id);
    }

    pub fn lookup(&self, virt_id: &VirtualCoordId) -> Option<PhysicalPageId> {
        self.mapping.get(virt_id).copied()
    }
}
