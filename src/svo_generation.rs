use bevy::math::IVec3;
use noise::{NoiseFn, Perlin};
use rayon::prelude::*;

use crate::octree::{SparseVoxelWorld, VoxelNode};

impl SparseVoxelWorld<i32> {
    pub fn new_from_noise(max_depth: u32, threshold: f64) -> Self {
        let perlin = Perlin::new(0);
        let scale = 0.05;

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
                        let nx = x as f64 * scale;
                        let ny = -y as f64 * scale;
                        let nz = z as f64 * scale;
                        let noise_val = perlin.get([nx, ny, nz]);

                        if noise_val > threshold {
                            Some(IVec3::new(x, y, z))
                        } else {
                            None
                        }
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
