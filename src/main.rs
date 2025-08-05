use bevy::{
    prelude::*,
    window::{PresentMode, WindowResolution},
};
use bevy_flycam::prelude::*;

use crate::octree::{SparseVoxelWorld, SvoEntity, SvoOctantsEntity, VOXEL_WORLD_DEPTH};

pub mod camera_controller;
pub mod octree;
pub mod svo_generation;

fn main() {
    // Remove unnecessary vulkan validation errors, PresentMode errors, Xsettings errors, and swapchain errors
    unsafe {
        std::env::set_var(
            "RUST_LOG",
            "warn,wgpu_hal::vulkan::instance=off,wgpu_hal::vulkan::conv=error,winit::platform_impl::linux::x11::xdisplay=error,bevy_render::view::window=error",
        );
    }

    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Octree Bevy".into(),
                        resolution: WindowResolution::new(1920., 1080.),
                        resizable: true,
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .insert_resource(MovementSettings {
            sensitivity: 0.00015, // default is 0.00012
            speed: 24.0,          // default is 12.0
        })
        .insert_resource(KeyBindings {
            move_descend: KeyCode::ControlLeft,
            ..Default::default()
        })
        .add_plugins(PlayerPlugin)
        .add_systems(PostStartup, setup)
        .add_systems(Update, update_svo)
        .run();
}

pub fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    camera: Query<&Transform, With<Camera3d>>,
) {
    // Light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            radius: 1.,
            range: 1000.,
            ..default()
        },
        Transform::from_xyz(0.0, 10.0, 5.0),
    ));

    let svo = SparseVoxelWorld::new_from_noise(VOXEL_WORLD_DEPTH, 0.5, 0);

    let camera_pos = camera.single().unwrap();

    // Generate mesh for SVO
    let svo_mesh = svo.generate_mesh(camera_pos.translation);
    commands.spawn((
        Mesh3d(meshes.add(svo_mesh)),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.0, 0.0),
        SvoEntity,
    ));

    let svo_octants_mesh = svo.generate_bounding_octants_mesh(camera_pos.translation);
    commands.spawn((
        Mesh3d(meshes.add(svo_octants_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
        SvoOctantsEntity,
    ));

    commands.insert_resource(svo);
}

#[allow(clippy::type_complexity)]
pub fn update_svo(
    svo: ResMut<SparseVoxelWorld<i32>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_queries: ParamSet<(
        Query<&mut Mesh3d, With<SvoEntity>>,
        Query<&mut Mesh3d, With<SvoOctantsEntity>>,
    )>,
    camera: Query<&Transform, With<Camera3d>>,
) {
    if let Ok(camera_pos) = camera.single() {
        // New SVO Mesh
        let new_mesh = svo.generate_mesh(camera_pos.translation);
        if let Ok(mut mesh_handle) = mesh_queries.p0().single_mut() {
            *mesh_handle = Mesh3d(meshes.add(new_mesh));
        }

        // New Octants SVO Mesh
        let new_octants_mesh = svo.generate_bounding_octants_mesh(camera_pos.translation);
        if let Ok(mut mesh_handle) = mesh_queries.p1().single_mut() {
            *mesh_handle = Mesh3d(meshes.add(new_octants_mesh));
        }
    }
}
