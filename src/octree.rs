use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};

pub const VOXEL_WORLD_SIZE: u32 = 8;
pub const LOD_DISTANCE: f32 = 50.;

#[derive(Debug)]
pub struct VoxelNode<T> {
    data: Option<T>,                          // Optional voxel data
    children: [Option<Box<VoxelNode<T>>>; 8], // Sparse children
    occupied: bool,
}

impl<T> VoxelNode<T> {
    pub fn new(data: Option<T>) -> Self {
        Self {
            children: Default::default(),
            occupied: data.is_some(),
            data,
        }
    }

    // fn is_empty(&self) -> bool {
    //     !self.occupied
    // }

    pub fn has_children(&self) -> bool {
        self.children.iter().any(|c| c.is_some())
    }
}

impl<T: Clone> VoxelNode<T> {
    fn insert(&mut self, x: i32, y: i32, z: i32, depth: u32, value: T) {
        self.occupied = true;

        if depth == 0 {
            self.data = Some(value);
            return;
        }

        let shift = depth - 1;

        // Determine which child octant the voxel belongs to
        let xi = if x >= 0 {
            ((x >> shift) & 1) as usize
        } else {
            (((x + 1) >> shift) & 1) as usize
        };
        let yi = if y >= 0 {
            ((y >> shift) & 1) as usize
        } else {
            (((y + 1) >> shift) & 1) as usize
        };
        let zi = if z >= 0 {
            ((z >> shift) & 1) as usize
        } else {
            (((z + 1) >> shift) & 1) as usize
        };

        let idx = (xi << 2) | (yi << 1) | zi;

        let child = self.children[idx].get_or_insert_with(|| Box::new(VoxelNode::new(None)));

        child.insert(x, y, z, depth - 1, value);
    }
}

#[derive(Resource)]
pub struct SparseVoxelWorld<T> {
    pub root: VoxelNode<T>,
    pub size: u32,
    pub max_depth: u32,
}

impl<T: Clone> SparseVoxelWorld<T> {
    pub fn insert(&mut self, x: i32, y: i32, z: i32, value: T) {
        self.root.insert(x, y, z, self.max_depth, value);
    }

    pub fn generate_mesh_from_svo(&self, camera_pos: [f32; 3]) -> Mesh {
        let mut positions_and_scales = Vec::new();

        // Recursive traversal to desired LOD depth
        #[allow(clippy::too_many_arguments)]
        fn traverse_lod<T>(
            node: &VoxelNode<T>,
            x: i32,
            y: i32,
            z: i32,
            depth: u32,
            camera_pos: [f32; 3],
            max_depth: u32,
            out: &mut Vec<([f32; 3], f32)>,
        ) {
            // Skip if empty branch
            if !node.occupied {
                return;
            }

            // Cube size in world units:
            // - depth == max_depth => 1.0
            // - every level up doubles the size
            let cube_size = 2f32.powi((max_depth - depth) as i32);

            let voxel_center = [
                x as f32 + cube_size * 0.5,
                y as f32 + cube_size * 0.5,
                z as f32 + cube_size * 0.5,
            ];

            // Distance from camera
            let dx = voxel_center[0] - camera_pos[0];
            let dy = voxel_center[1] - camera_pos[1];
            let dz = voxel_center[2] - camera_pos[2];
            let distance_sq = dx * dx + dy * dy + dz * dz;

            // Determine LOD based on distance
            // Example: farther voxels use lower depth
            let distance = distance_sq.sqrt();
            let lod_threshold = LOD_DISTANCE * (depth + 1) as f32;
            let should_stop = depth >= max_depth || distance > lod_threshold;

            // If reached target depth or this is a leaf
            if should_stop || !node.has_children() {
                out.push(([x as f32, y as f32, z as f32], cube_size));
                return;
            }

            let step = cube_size as i32 / 2; // half size in this level

            for (i, child) in node.children.iter().enumerate() {
                if let Some(child) = child {
                    // Decode child index bits
                    let xi = ((i >> 2) & 1) as i32;
                    let yi = ((i >> 1) & 1) as i32;
                    let zi = (i & 1) as i32;

                    // Calculate child position offsets:
                    // For signed space, child position = x + (xi * step) - half_step
                    // Where half_step = step, since step = half cube size at this level
                    // But since x is the min corner of parent cube,
                    // and we want the children to cover the full parent cube,
                    // the offsets become:

                    let child_x = x + (xi * step);
                    let child_y = y + (yi * step);
                    let child_z = z + (zi * step);

                    traverse_lod(child, child_x, child_y, child_z, depth + 1, camera_pos, max_depth, out);
                }
            }
        }

        // Adjust root call to start from negative half of world size so octree is centered at zero
        let half_world_size = 1 << self.max_depth;
        let root_min_corner = -half_world_size / 2;

        traverse_lod(
            &self.root,
            root_min_corner,
            root_min_corner,
            root_min_corner,
            0,
            camera_pos,
            self.max_depth,
            &mut positions_and_scales,
        );

        fn vertices_and_indices_from_voxel_positions(voxels: &[([f32; 3], f32)]) -> (Vec<[f32; 3]>, Vec<u32>) {
            // Vertices of a unit cube at origin
            const CUBE_VERTICES: [[f32; 3]; 8] = [
                [0.0, 0.0, 0.0], // 0
                [1.0, 0.0, 0.0], // 1
                [1.0, 1.0, 0.0], // 2
                [0.0, 1.0, 0.0], // 3
                [0.0, 0.0, 1.0], // 4
                [1.0, 0.0, 1.0], // 5
                [1.0, 1.0, 1.0], // 6
                [0.0, 1.0, 1.0], // 7
            ];

            // Indices for triangles (two per face)
            const CUBE_INDICES: [u32; 36] = [
                0, 2, 1, 0, 3, 2, // Bottom
                4, 5, 6, 4, 6, 7, // Top
                0, 1, 5, 0, 5, 4, // Front
                1, 2, 6, 1, 6, 5, // Right
                2, 3, 7, 2, 7, 6, // Back
                3, 0, 4, 3, 4, 7, // Left
            ];

            let mut vertices = Vec::new();
            let mut indices = Vec::new();

            for (i, &(pos, scale)) in voxels.iter().enumerate() {
                let base_index = (i * 8) as u32;

                // Add scaled cube vertices, offset by voxel position
                for &corner in &CUBE_VERTICES {
                    vertices.push([
                        pos[0] + corner[0] * scale,
                        pos[1] + corner[1] * scale,
                        pos[2] + corner[2] * scale,
                    ]);
                }

                // Add cube indices offset by base_index
                for &idx in &CUBE_INDICES {
                    indices.push(base_index + idx);
                }
            }

            (vertices, indices)
        }

        let (vertices, indices) = vertices_and_indices_from_voxel_positions(&positions_and_scales);

        fn generate_normals(voxel_count: usize) -> Vec<[f32; 3]> {
            fn normalise(v: [f32; 3]) -> [f32; 3] {
                let mag = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if mag > 0.0 { [v[0] / mag, v[1] / mag, v[2] / mag] } else { v }
            }

            const CUBE_NORMALS_PER_VERTEX: [[f32; 3]; 8] = [
                [-1.0, -1.0, -1.0], // 0
                [1.0, -1.0, -1.0],  // 1
                [1.0, 1.0, -1.0],   // 2
                [-1.0, 1.0, -1.0],  // 3
                [-1.0, -1.0, 1.0],  // 4
                [1.0, -1.0, 1.0],   // 5
                [1.0, 1.0, 1.0],    // 6
                [-1.0, 1.0, 1.0],   // 7
            ];

            let mut normals = Vec::with_capacity(voxel_count * CUBE_NORMALS_PER_VERTEX.len());

            for _ in 0..voxel_count {
                for &normal in &CUBE_NORMALS_PER_VERTEX {
                    normals.push(normalise(normal));
                }
            }

            normals
        }

        println!("{positions_and_scales:?}");

        let normals = generate_normals(positions_and_scales.len());

        // Create the Mesh
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            // Insert the vertex positions, indices, and normals
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(vertices))
            .with_inserted_indices(Indices::U32(indices))
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(normals))
    }
}
