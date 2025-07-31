use bevy::{
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
};
use std::sync::Arc;

use crate::octree::{Voxel, VoxelNode};

pub mod octree;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(ImagePlugin::default_nearest()).set(WindowPlugin {
                primary_window: Some(Window {
                    title: "SVO".into(),
                    resolution: (1920., 1080.).into(),
                    resizable: true,
                    ..default()
                }),
                ..default()
            }),
            WireframePlugin::default(),
        ))
        .insert_resource(WireframeConfig {
            global: false, // if true, renders all meshes as wireframe
            ..default()
        })
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    // Camera
    commands.spawn((
        Camera3d { ..default() },
        Transform::from_xyz(5.0, 5.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        GlobalTransform::default(),
    ));

    // Light
    commands.spawn((
        PointLight {
            intensity: 2000.0,
            range: 100.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, -10.0),
        GlobalTransform::default(),
    ));

    // Build and fill SVO (same as earlier)
    let mut root = Arc::new(VoxelNode::new());
    VoxelNode::insert(&mut root, IVec3::new(0, 0, 0), 4, Color::linear_rgb(1.0, 0.0, 0.0));
    VoxelNode::insert(&mut root, IVec3::new(0, 1, 0), 3, Color::linear_rgb(0.0, 1.0, 0.0));
    VoxelNode::insert(&mut root, IVec3::new(0, 2, 0), 3, Color::linear_rgb(0.0, 1.0, 1.0));

    let mut voxels = Vec::new();
    root.collect_leaf_voxels(4, IVec3::ZERO, 8, &mut voxels);

    // Create cube mesh
    let cube = Cuboid::default();
    let mesh_handle = meshes.add(Mesh::from(cube.mesh()));

    // Spawn voxel entities
    for Voxel { pos, colour } in voxels {
        commands.spawn((
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: colour,
                ..default()
            })),
            Transform::from_translation(pos.as_vec3()),
            GlobalTransform::default(),
        ));
    }
}
