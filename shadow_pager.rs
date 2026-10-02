use crate::page_table::{PhysicalPageId, TopologicalBlockTable, VirtualCoordId};
use std::collections::{HashMap, VecDeque};

pub struct ShadowPager {
    pub main_table: TopologicalBlockTable,
    pub shadow_updates: HashMap<VirtualCoordId, PhysicalPageId>,
    free_pages: VecDeque<PhysicalPageId>,
}

impl ShadowPager {
    pub fn new(total_physical_pages: usize) -> Self {
        let mut free_pages = VecDeque::new();
        for page_id in 0..total_physical_pages {
            free_pages.push_back(page_id);
        }
        Self {
            main_table: TopologicalBlockTable::new(),
            shadow_updates: HashMap::new(),
            free_pages,
        }
    }

    pub fn stage_shadow_update(&mut self, virt_id: VirtualCoordId) -> Result<PhysicalPageId, &'static str> {
        let shadow_phys = self.free_pages.pop_front().ok_or("VRAM Page Exhaustion")?;
        self.shadow_updates.insert(virt_id, shadow_phys);
        Ok(shadow_phys)
    }

    pub fn commit(&mut self) {
        for (virt_id, new_phys_id) in self.shadow_updates.drain() {
            if let Some(old_phys_id) = self.main_table.lookup(&virt_id) {
                self.free_pages.push_back(old_phys_id);
            }
            self.main_table.map(virt_id, new_phys_id);
        }
    }

    pub fn rollback(&mut self) {
        for (_, shadow_phys_id) in self.shadow_updates.drain() {
            self.free_pages.push_back(shadow_phys_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cow_commit_and_rollback() {
        let mut pager = ShadowPager::new(10);
        let virt_coord = (1, 2);

        // Stage speculative update
        let _staged = pager.stage_shadow_update(virt_coord).unwrap();
        assert_eq!(pager.main_table.lookup(&virt_coord), None);

        // Rollback drops shadow page
        pager.rollback();
        assert_eq!(pager.main_table.lookup(&virt_coord), None);

        // Stage and Commit
        let staged_again = pager.stage_shadow_update(virt_coord).unwrap();
        pager.commit();
        assert_eq!(pager.main_table.lookup(&virt_coord), Some(staged_again));
    }

    #[test]
    fn test_page_exhaustion() {
        let mut pager = ShadowPager::new(1);
        assert!(pager.stage_shadow_update((0, 0)).is_ok());
        assert_eq!(pager.stage_shadow_update((0, 1)), Err("VRAM Page Exhaustion"));
    }
}
