use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};

const VOXEL_WORLD_SIZE: u32 = 8;

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

        let xi = ((x >> shift) & 1) as usize;
        let yi = ((y >> shift) & 1) as usize;
        let zi = ((z >> shift) & 1) as usize;

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

    pub fn generate_mesh_from_svo(&self, target_depth: u32) -> Mesh {
        // Ensure that target_depth is not too large
        assert!(
            target_depth <= self.max_depth,
            "Tried to traverse to a depth bigger than maximum depth of SVO: {target_depth}"
        );

        let mut positions_and_scales = Vec::new();

        // Recursive traversal to desired LOD depth
        #[allow(clippy::too_many_arguments)]
        fn traverse_lod<T>(
            node: &VoxelNode<T>,
            x: i32,
            y: i32,
            z: i32,
            depth: u32,
            target_depth: u32,
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

            // If reached target depth or this is a leaf
            if depth == target_depth || !node.has_children() {
                out.push(([x as f32, y as f32, z as f32], cube_size));
                return;
            }

            let step = cube_size as i32 / 2; // half size in this level

            for (i, child) in node.children.iter().enumerate() {
                if let Some(child) = child {
                    let dx = ((i as i32 >> 2) & 1) * step;
                    let dy = ((i as i32 >> 1) & 1) * step;
                    let dz = (i as i32 & 1) * step;

                    traverse_lod(child, x + dx, y + dy, z + dz, depth + 1, target_depth, max_depth, out);
                }
            }
        }

        traverse_lod(
            &self.root,
            0,
            0,
            0,
            0,
            target_depth,
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

pub fn setup_voxel_world(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut svo = SparseVoxelWorld {
        root: VoxelNode::new(None),
        size: VOXEL_WORLD_SIZE,
        max_depth: 3,
    };

    svo.insert(0, 0, 0, 1);
    svo.insert(1, 0, 0, 2);
    svo.insert(4, 0, 0, 2);
    // svo.root.insert(0, 1, 0, 2, 2);
    // svo.root.insert(0, 4, 0, 1, 1);

    // Generate mesh for SVO
    let svo_mesh = svo.generate_mesh_from_svo(svo.max_depth);

    commands.spawn((
        Mesh3d(meshes.add(svo_mesh)),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    commands.insert_resource(svo);
}
