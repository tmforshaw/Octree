use bevy::math::IVec3;
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;

use crate::octree::{SparseVoxelWorld, VoxelNode};

impl SparseVoxelWorld<i32> {
    pub fn new_from_noise(max_depth: u32, threshold: f64, scale: f64, height_scale: f64) -> Self {
        let perlin = Perlin::new(0);

        let size = 2i32.pow(max_depth);
        let range = -size / 2..size / 2;

        // Flatten x, y, z coordinates in parallel
        let positions = range
            .clone()
            .into_par_iter()
            .flat_map_iter(|x| {
                let range_clone = range.clone();

                range.clone().flat_map(move |y| {
                    range_clone.clone().filter_map(move |z| {
                        // Parameters to tweak
                        let terrain_freq = 0.01; // Frequency for base terrain
                        let detail_freq = 0.05; // Frequency for overhangs and erosion
                        let height_scale = 32.0; // Maximum terrain height
                        let erosion_strength = 0.5; // How strongly erosion influences the terrain

                        // Scale positions
                        let nx = x as f64 * terrain_freq;
                        let nz = z as f64 * terrain_freq;
                        let ny = y as f64 * terrain_freq; // For erosion & caves

                        // --- Base heightmap ---
                        let base_height = perlin.get([nx, nz]) * height_scale;

                        // --- Erosion: fine noise modifies terrain surface subtly ---
                        let erosion = perlin.get([nx * 2.0, nz * 2.0]) * erosion_strength * height_scale;
                        let terrain_height = base_height + erosion;

                        // --- Overhangs / detail noise (adds cliffs and floating shapes) ---
                        let overhang_noise = perlin.get([x as f64 * detail_freq, y as f64 * detail_freq, z as f64 * detail_freq]);

                        // This shifts terrain surface vertically to create overhangs
                        let terrain_cutoff = terrain_height + (overhang_noise * 6.0); // adjust 6.0 for cliffiness

                        // --- Final solidity check ---
                        let solid = y as f64 <= terrain_cutoff;

                        if solid { Some(IVec3::new(x, y, z)) } else { None }
                    })
                })
            })
            .collect::<Vec<_>>();

        let mut svo = Self {
            root: VoxelNode::new(None),
            max_depth,
        };

        // Sequential insertion to avoid thread-safety issues
        for pos in positions {
            svo.insert(pos, pos.y / 2i32.pow(svo.max_depth));
        }

        svo
    }
}
