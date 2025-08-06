use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use num_traits::AsPrimitive;
use rayon::prelude::*;

pub const VOXEL_WORLD_DEPTH: u32 = 8;
pub const LOD_DISTANCE: f32 = 20.;
pub const SHOW_OCTANTS_MESH: bool = false;

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

impl<T: Clone + Sync> VoxelNode<T> {
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

impl<T: Clone + Send + Sync + AsPrimitive<f32>> SparseVoxelWorld<T> {
    pub fn get_root_origin(&self) -> IVec3 {
        // Centre the origin at (0, 0, 0)
        let world_size = 1 << self.max_depth;
        IVec3::splat(-world_size / 2)
    }

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
    ) -> Vec<(Vec3, f32, Option<T>)> {
        // Skip empty nodes
        if !node.occupied {
            return Vec::new();
        }

        let size = 2i32.pow(max_depth - depth);
        let size_f = size as f32;

        let should_stop = {
            // LOD distance check
            let voxel_centre = origin.as_vec3() + size_f * 0.5;
            let distance = voxel_centre.distance(camera_pos);
            let lod_threshold = LOD_DISTANCE * size_f;

            depth >= max_depth || distance > lod_threshold
        };

        // Leaf node or LOD cutoff
        if should_stop || !node.has_children() {
            return vec![(origin.as_vec3(), size_f, node.data)];
        }

        // Half size for child nodes
        let half = size / 2;
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

        // Parallel traversal over 8 children
        node.children
            .par_iter()
            .zip(OFFSETS)
            .filter_map(|(child, offset)| {
                child.as_ref().map(|child_node| {
                    let child_origin = origin + offset * half;

                    Self::traverse_lod(child_node, child_origin, depth + 1, camera_pos, max_depth)
                })
            })
            .reduce(Vec::new, |mut acc, mut chunk| {
                acc.append(&mut chunk);

                acc
            })
    }

    pub fn generate_mesh(&self, camera_pos: Vec3) -> Mesh {
        let positions_and_scales = Self::traverse_lod(
            &self.root,
            self.get_root_origin(),
            0, // start depth
            camera_pos,
            self.max_depth,
        );

        #[allow(clippy::identity_op, clippy::type_complexity)]
        fn vertices_indices_and_normals_from_voxels<T>(
            voxels: &[(Vec3, f32, Option<T>)],
        ) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>)
        where
            T: Clone + Send + Sync + AsPrimitive<f32>,
        {
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

            let num_voxels = voxels.len();
            let vertices_per_cube = 24;
            let indices_per_cube = 36;

            let total_vertices = num_voxels * vertices_per_cube;
            let total_indices = num_voxels * indices_per_cube;

            let mut vertices = vec![[0.; 3]; total_vertices];
            let mut normals = vec![[0.; 3]; total_vertices];
            let mut colours = vec![[0.; 4]; total_vertices];
            let mut indices = vec![0; total_indices];

            // Iterate over chunks of vertices, normals, and indices to generate them in parallel
            vertices
                .par_chunks_mut(vertices_per_cube)
                .zip(normals.par_chunks_mut(vertices_per_cube))
                .zip(colours.par_chunks_mut(vertices_per_cube))
                .zip(indices.par_chunks_mut(indices_per_cube))
                .enumerate()
                .for_each(|(i, (((v_chunk, n_chunk), c_chunk), i_chunk))| {
                    let (pos, scale, data) = voxels[i];

                    let base_index = (i * 24) as u32; // 24 vertices per cube

                    let colour = if let Some(data) = data {
                        Color::hsv(data.as_() * 255., 1.0, 1.)
                    } else {
                        Color::srgb_u8(124, 144, 255)
                    }
                    .to_linear()
                    .to_f32_array();

                    for (face_id, face) in FACE_OFFSETS.iter().enumerate() {
                        // Push 4 vertices for this face
                        for (corner_id, &corner) in face.iter().enumerate() {
                            let v_idx = face_id * 4 + corner_id;

                            v_chunk[v_idx] = (pos + corner * scale).to_array();
                            n_chunk[v_idx] = FACE_NORMALS[face_id];
                            c_chunk[v_idx] = colour;
                        }

                        let i_idx = face_id * 6;
                        // Two triangles per face (quad)
                        i_chunk[i_idx..i_idx + 6].copy_from_slice(&[
                            base_index + (face_id * 4 + 0) as u32,
                            base_index + (face_id * 4 + 1) as u32,
                            base_index + (face_id * 4 + 2) as u32,
                            base_index + (face_id * 4 + 0) as u32,
                            base_index + (face_id * 4 + 2) as u32,
                            base_index + (face_id * 4 + 3) as u32,
                        ]);
                    }
                });

            (vertices, normals, colours, indices)
        }

        let (vertices, normals, colours, indices) = vertices_indices_and_normals_from_voxels(&positions_and_scales);

        // Create the Mesh
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            // Insert the vertex positions, indices, and normals
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(vertices))
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colours))
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, VertexAttributeValues::Float32x3(normals))
            .with_inserted_indices(Indices::U32(indices))
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

        pub fn collect_octree_lines<T: Sync>(
            node: &VoxelNode<T>,
            origin: Vec3,
            depth: u32,
            max_depth: u32,
            camera_pos: Vec3,
        ) -> (Vec<[f32; 3]>, Vec<[f32; 4]>) {
            let size = 2u32.pow(max_depth - depth) as f32;
            let centre = origin + size * 0.5;

            let should_stop = {
                // LOD distance check
                let distance = centre.distance(camera_pos);
                let lod_threshold = LOD_DISTANCE * size;

                depth >= max_depth || distance > lod_threshold
            };

            // If reached leaf or LOD cutoff
            if should_stop || !node.occupied {
                return (Vec::new(), Vec::new());
            }

            // Compute cube corners in world space
            let corners: Vec<[f32; 3]> = CUBE_CORNERS
                .iter()
                .map(|c| {
                    let line_vertex = origin + c * size;
                    let centre_dir = (line_vertex - centre).normalize();
                    (line_vertex + centre_dir * 0.001 * size).to_array()
                })
                .collect();

            let colour = Color::hsv(depth as f32 / max_depth as f32 * 255., 1.0, 1.)
                .to_linear()
                .to_f32_array();

            // Current node edges
            let mut lines = Vec::with_capacity(CUBE_EDGES.len() * 2);
            let mut colours = Vec::with_capacity(CUBE_EDGES.len() * 2);

            for &(a, b) in &CUBE_EDGES {
                lines.push(corners[a]);
                lines.push(corners[b]);
                colours.push(colour);
                colours.push(colour);
            }

            // Recurse into children
            if node.has_children() {
                let half = size * 0.5;
                use rayon::prelude::*;

                let children_results: Vec<_> = node
                    .children
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, child)| {
                        child.as_ref().map(|child| {
                            let offset = origin + Vec3::new(((i >> 2) & 1) as f32, ((i >> 1) & 1) as f32, (i & 1) as f32) * half;

                            collect_octree_lines(child, offset, depth + 1, max_depth, camera_pos)
                        })
                    })
                    .collect();

                // Merge children results
                for (mut child_lines, mut child_colours) in children_results {
                    lines.append(&mut child_lines);
                    colours.append(&mut child_colours);
                }
            }

            (lines, colours)
        }

        let (lines, colours) = collect_octree_lines(&self.root, self.get_root_origin().as_vec3(), 0, self.max_depth, camera_pos);

        // Create the Mesh
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::RENDER_WORLD)
            // Insert the lines
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(lines))
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, VertexAttributeValues::Float32x4(colours))
    }
}

#[derive(Resource)]
pub struct SvoUpdateState {
    pub last_camera_pos: Vec3,
    pub last_lod_region: IVec3,
}

#[allow(clippy::type_complexity)]
pub fn update_svo(
    mut state: ResMut<SvoUpdateState>,
    svo: ResMut<SparseVoxelWorld<i32>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_queries: ParamSet<(
        Query<&mut Mesh3d, With<SvoEntity>>,
        Query<&mut Mesh3d, With<SvoOctantsEntity>>,
    )>,
    camera_query: Query<&Transform, With<Camera3d>>,
) {
    let camera = camera_query.single();

    if let Ok(camera) = camera {
        let camera_pos = camera.translation;

        // Convert position to a "LOD region" (like chunking)
        let root_cell_size = 2u32.pow(svo.max_depth) as f32;
        let lod_cell_size = root_cell_size.min(LOD_DISTANCE * 0.5);
        let current_region = IVec3::new(
            (camera_pos.x / lod_cell_size).floor() as i32,
            (camera_pos.y / lod_cell_size).floor() as i32,
            (camera_pos.z / lod_cell_size).floor() as i32,
        );

        // Only regenerate mesh if camera moved into a new region
        if current_region != state.last_lod_region {
            // New SVO Mesh
            let new_mesh = svo.generate_mesh(camera_pos);
            if let Ok(mut mesh_handle) = mesh_queries.p0().single_mut() {
                *mesh_handle = Mesh3d(meshes.add(new_mesh));
            }

            if SHOW_OCTANTS_MESH {
                // New Octants SVO Mesh
                let new_octants_mesh = svo.generate_bounding_octants_mesh(camera_pos);
                if let Ok(mut mesh_handle) = mesh_queries.p1().single_mut() {
                    *mesh_handle = Mesh3d(meshes.add(new_octants_mesh));
                }
            }

            // Update state
            state.last_camera_pos = camera_pos;
            state.last_lod_region = current_region;
        }
    }
}
