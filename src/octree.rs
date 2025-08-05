use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};

pub const VOXEL_WORLD_DEPTH: u32 = 7;
pub const LOD_DISTANCE: f32 = 10.;

#[derive(Component)]
pub struct SvoEntity;

#[derive(Component)]
pub struct SvoOctantsEntity;

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
    pub max_depth: u32,
}

impl<T: Clone> SparseVoxelWorld<T> {
    pub fn insert(&mut self, pos: IVec3, value: T) {
        // Centre the origin at (0, 0, 0)
        let world_size = 1 << self.max_depth;
        let root_min_corner = IVec3::splat(-world_size / 2);

        self.root.insert(pos, self.max_depth, root_min_corner, value);
    }

    // Returns Vec of node centres and their size
    pub fn traverse_lod(
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
        let lod_threshold = LOD_DISTANCE * max_depth as f32;
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

                Self::traverse_lod(child, child_origin, depth + 1, camera_pos, max_depth, out);
            }
        }
    }

    pub fn generate_mesh(&self, camera_pos: Vec3) -> Mesh {
        let mut positions_and_scales = Vec::new();

        // Centre the world at (0,0,0)
        let world_size = 1 << self.max_depth;
        let root_min = IVec3::splat(-world_size / 2);

        Self::traverse_lod(
            &self.root,
            root_min,
            0, // start depth
            camera_pos,
            self.max_depth,
            &mut positions_and_scales,
        );

        #[allow(clippy::identity_op)]
        fn vertices_indices_and_normals_from_voxels(voxels: &[(Vec3, f32)]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
            // Offsets for all 6 cube faces
            const FACE_OFFSETS: [[Vec3; 4]; 6] = [
                // Bottom (-Y)
                [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                ],
                // Top (+Y)
                [
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(0.0, 1.0, 1.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(1.0, 1.0, 0.0),
                ],
                // Front (-Z)
                [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                ],
                // Back (+Z)
                [
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(1.0, 0.0, 1.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(0.0, 1.0, 1.0),
                ],
                // Left (-X)
                [
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 1.0, 1.0),
                    Vec3::new(0.0, 1.0, 0.0),
                ],
                // Right (+X)
                [
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(1.0, 1.0, 0.0),
                    Vec3::new(1.0, 1.0, 1.0),
                    Vec3::new(1.0, 0.0, 1.0),
                ],
            ];

            // Normals for all 6 cube faces
            const FACE_NORMALS: [[f32; 3]; 6] = [
                [0.0, -1.0, 0.0], // Bottom
                [0.0, 1.0, 0.0],  // Top
                [0.0, 0.0, -1.0], // Front
                [0.0, 0.0, 1.0],  // Back
                [-1.0, 0.0, 0.0], // Left
                [1.0, 0.0, 0.0],  // Right
            ];

            let mut vertices = Vec::new();
            let mut normals = Vec::new();
            let mut indices = Vec::new();

            for (i, &(pos, scale)) in voxels.iter().enumerate() {
                let base_index = (i * 24) as u32; // 24 vertices per cube

                for (face_id, face) in FACE_OFFSETS.iter().enumerate() {
                    // Push 4 vertices for this face
                    for &corner in face {
                        vertices.push((pos + corner * scale).to_array());
                        normals.push(FACE_NORMALS[face_id]);
                    }

                    // Two triangles per face (quad)
                    indices.extend_from_slice(&[
                        base_index + (face_id * 4 + 0) as u32,
                        base_index + (face_id * 4 + 1) as u32,
                        base_index + (face_id * 4 + 2) as u32,
                        base_index + (face_id * 4 + 0) as u32,
                        base_index + (face_id * 4 + 2) as u32,
                        base_index + (face_id * 4 + 3) as u32,
                    ]);
                }
            }

            (vertices, normals, indices)
        }

        let (vertices, normals, indices) = vertices_indices_and_normals_from_voxels(&positions_and_scales);

        // Create the Mesh
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            // Insert the vertex positions, indices, and normals
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(vertices))
            .with_inserted_indices(Indices::U32(indices))
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(normals))
    }

    pub fn generate_bounding_octants_mesh(&self, camera_pos: Vec3) -> Mesh {
        const CUBE_CORNERS: [Vec3; 8] = [
            Vec3::new(0.0, 0.0, 0.0), // 0
            Vec3::new(1.0, 0.0, 0.0), // 1
            Vec3::new(1.0, 1.0, 0.0), // 2
            Vec3::new(0.0, 1.0, 0.0), // 3
            Vec3::new(0.0, 0.0, 1.0), // 4
            Vec3::new(1.0, 0.0, 1.0), // 5
            Vec3::new(1.0, 1.0, 1.0), // 6
            Vec3::new(0.0, 1.0, 1.0), // 7
        ];

        const CUBE_EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            // bottom loop
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            // top loop
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            // vertical edges
        ];

        fn collect_octree_lines<T>(
            node: &VoxelNode<T>,
            origin: Vec3,
            depth: u32,
            max_depth: u32,
            camera_pos: Vec3,
            lines: &mut Vec<[f32; 3]>,
            colours: &mut Vec<[f32; 4]>,
        ) {
            let size = 2u32.pow(max_depth - depth) as f32;
            let centre = origin + size * 0.5;

            // LOD distance check
            let distance = centre.distance(camera_pos);
            let lod_threshold = LOD_DISTANCE * max_depth as f32;
            let should_stop = depth >= max_depth || distance > lod_threshold;

            // If reached leaf or LOD cutoff
            if should_stop || !node.occupied {
                return;
            }

            // Compute the cube corners in world space
            let corners: Vec<[f32; 3]> = CUBE_CORNERS
                .iter()
                .map(|c| {
                    // Calculate the line vertex
                    let line_vertex = origin + c * size;

                    // Calculate the distance to the node centre from this line
                    let centre_dir = (line_vertex - centre).normalize();

                    // Push the vertex away from the centre by a small amount
                    (line_vertex + centre_dir * 0.001 * size).to_array()
                })
                .collect();

            let colour = Color::hsv(depth as f32 / max_depth as f32 * 255., 1.0, 1.)
                .to_linear()
                .to_f32_array();

            // Push the edges as line segments (2 points per line)
            for &(a, b) in &CUBE_EDGES {
                lines.push(corners[a]); // Start
                lines.push(corners[b]); // End

                colours.push(colour);
                colours.push(colour);
            }

            // Recurse into children if present
            if node.has_children() {
                let half = size * 0.5;
                for (i, child) in node.children.iter().enumerate() {
                    if let Some(child) = child {
                        let offset = origin + Vec3::new(((i >> 2) & 1) as f32, ((i >> 1) & 1) as f32, (i & 1) as f32) * half;
                        collect_octree_lines(child, offset, depth + 1, max_depth, camera_pos, lines, colours);
                    }
                }
            }
        }

        // Centre world around (0,0,0)
        let root_size = 2f32.powi(self.max_depth as i32);
        let root_origin = Vec3::splat(-root_size * 0.5);

        let (mut lines, mut colours) = (Vec::new(), Vec::new());
        collect_octree_lines(
            &self.root,
            root_origin,
            0,
            self.max_depth,
            camera_pos,
            &mut lines,
            &mut colours,
        );

        // Create the Mesh
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            // Insert the lines
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(lines))
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colours))
    }
}
