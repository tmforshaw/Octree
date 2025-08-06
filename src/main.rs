use bevy::{
    prelude::*,
    window::{PresentMode, WindowResolution},
};
use bevy_flycam::prelude::*;

use crate::octree::{
    SHOW_OCTANTS_MESH, SparseVoxelWorld, SvoEntity, SvoOctantsEntity, SvoUpdateState, VOXEL_WORLD_DEPTH, update_svo,
};

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
        DirectionalLight {
            illuminance: 400.,
            shadows_enabled: false,
            ..default()
        },
        Transform::from_xyz(0.0, 1000.0, 1000.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let svo = SparseVoxelWorld::new_from_noise(VOXEL_WORLD_DEPTH, 0.5, 0);

    let camera_pos = camera.single().unwrap().translation;

    // Generate mesh for SVO
    let svo_mesh = svo.generate_mesh(camera_pos);
    commands.spawn((
        Mesh3d(meshes.add(svo_mesh)),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.0, 0.0),
        SvoEntity,
    ));

    if SHOW_OCTANTS_MESH {
        let svo_octants_mesh = svo.generate_bounding_octants_mesh(camera_pos);
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
    }

    commands.insert_resource(svo);

    // Initial SVO state
    commands.insert_resource(SvoUpdateState {
        last_camera_pos: camera_pos,
        last_lod_region: IVec3::splat(0),
    });
}
