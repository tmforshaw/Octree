use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};

pub const VOXEL_WORLD_SIZE: u32 = 8;
pub const LOD_DISTANCE: f32 = 50.;

#[derive(Debug)]
pub struct VoxelNode<T> {
    data: Option<T>,
    children: [Option<Box<VoxelNode<T>>>; 8],
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

    pub fn has_children(&self) -> bool {
        self.children.iter().any(|c| c.is_some())
    }
}

impl<T: Clone> VoxelNode<T> {
    fn insert(&mut self, pos: IVec3, depth: u32, mut origin: IVec3, value: T) {
        self.occupied = true;

        // Leaf node when bottom is reached
        if depth == 0 {
            self.data = Some(value);
            return;
        }

        // Double the cube size until at the current depth's size (Size == 1 for leaf)
        let half_size = 1 << (depth - 1);
        let centre = origin + IVec3::splat(half_size);

        // Calculate the index of this position, and offset the current centre
        let mut idx = 0;
        if pos.x >= centre.x {
            idx |= 0b100;
            origin.x += half_size;
        }
        if pos.y >= centre.y {
            idx |= 0b010;
            origin.y += half_size;
        }
        if pos.z >= centre.z {
            idx |= 0b001;
            origin.z += half_size;
        }

        // Get or create a new child for this node at idx, then insert into it
        let child = self.children[idx].get_or_insert_with(|| Box::new(VoxelNode::new(None)));
        child.insert(pos, depth - 1, origin, value);
    }
}

#[derive(Resource)]
pub struct SparseVoxelWorld<T> {
    pub root: VoxelNode<T>,
    pub size: u32,
    pub max_depth: u32,
}

impl<T: Clone> SparseVoxelWorld<T> {
    pub fn insert(&mut self, pos: IVec3, value: T) {
        // Centre the origin at (0, 0, 0)
        let world_size = 1 << self.max_depth;
        let root_min_corner = IVec3::splat(-world_size / 2);

        self.root.insert(pos, self.max_depth, root_min_corner, value);
    }

    pub fn generate_mesh_from_svo(&self, camera_pos: Vec3) -> Mesh {
        let mut positions_and_scales = Vec::new();

        #[allow(clippy::too_many_arguments)]
        fn traverse_lod<T>(
            node: &VoxelNode<T>,
            origin: IVec3,
            depth: u32,
            camera_pos: Vec3,
            max_depth: u32,
            out: &mut Vec<(Vec3, f32)>,
        ) {
            // Skip empty nodes
            if !node.occupied {
                return;
            }

            // Cube size in world units
            let cube_size = 2i32.pow(max_depth - depth);
            let cube_size_f = cube_size as f32;

            // Compute the voxel centre for LOD check
            let voxel_center = origin.as_vec3() + Vec3::splat(cube_size_f * 0.5);

            // LOD distance check
            let distance = voxel_center.distance(camera_pos);
            let lod_threshold = LOD_DISTANCE * (depth + 1) as f32;
            let should_stop = depth >= max_depth || distance > lod_threshold;

            // If reached leaf or LOD cutoff
            if should_stop || !node.has_children() {
                out.push((origin.as_vec3(), cube_size_f));
                return;
            }

            // Half size for child nodes
            let half = cube_size / 2;

            // Child octant offsets (8 combinations of x,y,z)
            const OFFSETS: [IVec3; 8] = [
                IVec3::new(0, 0, 0),
                IVec3::new(0, 0, 1),
                IVec3::new(0, 1, 0),
                IVec3::new(0, 1, 1),
                IVec3::new(1, 0, 0),
                IVec3::new(1, 0, 1),
                IVec3::new(1, 1, 0),
                IVec3::new(1, 1, 1),
            ];

            // Traverse into all children of this node
            for (child, offset) in node.children.iter().zip(OFFSETS) {
                if let Some(child) = child {
                    let child_origin = origin + offset * half;

                    traverse_lod(child, child_origin, depth + 1, camera_pos, max_depth, out);
                }
            }
        }

        // Centre the world at (0,0,0)
        let world_size = 1 << self.max_depth;
        let root_min = IVec3::splat(-world_size / 2);

        traverse_lod(
            &self.root,
            root_min,
            0, // start depth
            camera_pos,
            self.max_depth,
            &mut positions_and_scales,
        );

        fn vertices_and_indices_from_voxel_positions(voxels: &[(Vec3, f32)]) -> (Vec<[f32; 3]>, Vec<u32>) {
            // Vertices of a cube centered at origin, ranging from -0.5 to 0.5
            const CUBE_VERTICES: [Vec3; 8] = [
                Vec3::new(-0.5, -0.5, -0.5), // 0
                Vec3::new(0.5, -0.5, -0.5),  // 1
                Vec3::new(0.5, 0.5, -0.5),   // 2
                Vec3::new(-0.5, 0.5, -0.5),  // 3
                Vec3::new(-0.5, -0.5, 0.5),  // 4
                Vec3::new(0.5, -0.5, 0.5),   // 5
                Vec3::new(0.5, 0.5, 0.5),    // 6
                Vec3::new(-0.5, 0.5, 0.5),   // 7
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
                // Add scaled cube vertices, offset by voxel position
                for &corner in &CUBE_VERTICES {
                    vertices.push((pos + corner * scale).to_array());
                }

                // Add cube indices offset by base_index
                let base_index = (i * 8) as u32;
                for &idx in &CUBE_INDICES {
                    indices.push(base_index + idx);
                }
            }

            (vertices, indices)
        }

        let (vertices, indices) = vertices_and_indices_from_voxel_positions(&positions_and_scales);

        fn generate_normals(voxel_count: usize) -> Vec<[f32; 3]> {
            fn normalise(v: Vec3) -> [f32; 3] {
                let mag = v.length();
                if mag > 0.0 { v / mag } else { v }.to_array()
            }

            const CUBE_NORMALS_PER_VERTEX: [Vec3; 8] = [
                Vec3::new(-1.0, -1.0, -1.0), // 0
                Vec3::new(1.0, -1.0, -1.0),  // 1
                Vec3::new(1.0, 1.0, -1.0),   // 2
                Vec3::new(-1.0, 1.0, -1.0),  // 3
                Vec3::new(-1.0, -1.0, 1.0),  // 4
                Vec3::new(1.0, -1.0, 1.0),   // 5
                Vec3::new(1.0, 1.0, 1.0),    // 6
                Vec3::new(-1.0, 1.0, 1.0),   // 7
            ];

            let mut normals = Vec::with_capacity(voxel_count * CUBE_NORMALS_PER_VERTEX.len());

            for _ in 0..voxel_count {
                for &normal in &CUBE_NORMALS_PER_VERTEX {
                    normals.push(normalise(normal));
                }
            }

            normals
        }

        println!("{:?}", positions_and_scales.iter().map(|(pos, _)| pos).collect::<Vec<_>>());

        let normals = generate_normals(positions_and_scales.len());

        // Create the Mesh
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            // Insert the vertex positions, indices, and normals
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(vertices))
            .with_inserted_indices(Indices::U32(indices))
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(normals))
    }
}
