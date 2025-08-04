use bevy::{
    prelude::*,
    window::{PresentMode, WindowResolution},
};

use crate::octree::{SparseVoxelWorld, VOXEL_WORLD_SIZE, VoxelNode};

pub mod octree;

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
        .add_systems(Startup, setup)
        .run();
}

pub fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    // light
    let light_pos = Transform::from_xyz(0.0, 10.0, 5.0);
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        light_pos,
    ));

    // commands.spawn((
    //     Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
    //     MeshMaterial3d(materials.add(Color::srgb_u8(255, 0, 0))),
    //     light_pos.with_scale(Vec3::splat(0.3)),
    // ));

    // camera
    let camera_pos = Transform::from_xyz(0.0, 10., 30.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((Camera3d::default(), camera_pos));

    let mut svo = SparseVoxelWorld {
        root: VoxelNode::new(None),
        size: VOXEL_WORLD_SIZE,
        max_depth: 3,
    };

    svo.insert(0, 0, 0, 1);
    svo.insert(1, 0, 0, 2);
    svo.insert(4, 0, 0, 2);
    svo.insert(0, 1, -20, 2);
    // svo.root.insert(0, 1, 0, 2, 2);
    // svo.root.insert(0, 4, 0, 1, 1);

    // Generate mesh for SVO
    let camera_pos = camera_pos.translation;
    let svo_mesh = svo.generate_mesh_from_svo([camera_pos.x, camera_pos.y, camera_pos.z]);

    commands.spawn((
        Mesh3d(meshes.add(svo_mesh)),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
    ));

    commands.insert_resource(svo);
}
