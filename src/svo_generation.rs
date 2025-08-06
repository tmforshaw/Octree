use bevy::math::IVec3;
use noise::{NoiseFn, Perlin};

use crate::octree::{SparseVoxelWorld, VoxelNode};

impl<T: Clone + Sync> SparseVoxelWorld<T> {
    pub fn new_from_noise(max_depth: u32, threshold: f64, value: T) -> Self {
        let perlin = Perlin::new(0);
        let scale = 0.1;

        let size = 2i32.pow(max_depth);

        let positions = ((-size / 2)..size / 2)
            .flat_map(|x| {
                ((-size / 2)..size / 2).flat_map(move |y| {
                    ((-size / 2)..size / 2).filter_map(move |z| {
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

        for pos in positions {
            svo.insert(pos, value.clone());
        }

        svo
    }
}
