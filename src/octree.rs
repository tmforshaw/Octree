use std::sync::Arc;

use bevy::prelude::*;

#[derive(Debug, Clone)]
pub struct Voxel {
    pub pos: IVec3,
    pub colour: Color,
}

#[derive(Debug, Clone)]
pub struct VoxelNode {
    pub children: [Option<Arc<VoxelNode>>; 8],
    pub is_leaf: bool,
    pub colour: Option<Color>,
}

impl Default for VoxelNode {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelNode {
    pub fn new() -> Self {
        Self {
            children: Default::default(),
            is_leaf: false,
            colour: None,
        }
    }

    /// Insert a voxel into the SVO at a given depth
    pub fn insert(root: &mut Arc<Self>, pos: IVec3, depth: u32, color: Color) {
        // Convert Arc<Self> to a mutable reference by cloning if needed
        // Arc itself is immutable, but for simplicity in this demo
        // we create a new node if required (functional-style updates)
        let root_mut = Arc::make_mut(root);

        if depth == 0 {
            root_mut.is_leaf = true;
            root_mut.colour = Some(color);
            return;
        }

        // Compute child index (XYZ to 0..7)
        let half_size = 1 << (depth - 1);
        let idx = ((pos.x >= half_size) as usize) << 2 | ((pos.y >= half_size) as usize) << 1 | ((pos.z >= half_size) as usize);

        // Create child if needed
        if root_mut.children[idx].is_none() {
            root_mut.children[idx] = Some(Arc::new(VoxelNode::new()));
        }

        // Compute new local position
        let new_pos = IVec3::new(
            pos.x.rem_euclid(half_size),
            pos.y.rem_euclid(half_size),
            pos.z.rem_euclid(half_size),
        );

        // Recursive insert
        VoxelNode::insert(root_mut.children[idx].as_mut().unwrap(), new_pos, depth - 1, color);
    }

    /// Recursively collect all leaf voxels with their world positions
    pub fn collect_leaf_voxels(&self, depth: u32, origin: IVec3, size: i32, voxels: &mut Vec<Voxel>) {
        if self.is_leaf || depth == 0 {
            if let Some(colour) = self.colour {
                voxels.push(Voxel { pos: origin, colour });
            }
            return;
        }

        let child_size = size / 2;
        for (i, child) in self.children.iter().enumerate() {
            if let Some(node) = child {
                let offset = IVec3::new(
                    if i & 4 != 0 { child_size } else { 0 },
                    if i & 2 != 0 { child_size } else { 0 },
                    if i & 1 != 0 { child_size } else { 0 },
                );

                node.collect_leaf_voxels(depth - 1, origin + offset, child_size, voxels);
            }
        }
    }
}
