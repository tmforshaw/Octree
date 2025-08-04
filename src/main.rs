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

    // Camera
    let camera_pos = Transform::from_xyz(0.0, 10., 20.0).looking_at(Vec3::ZERO, Vec3::Y);
    commands.spawn((Camera3d::default(), camera_pos));

    let mut svo = SparseVoxelWorld {
        root: VoxelNode::new(None),
        size: VOXEL_WORLD_SIZE,
        max_depth: 3,
    };

    for i in -(2_i32.pow(svo.max_depth)) / 2..(2_i32.pow(svo.max_depth)) / 2 {
        for j in -(2_i32.pow(svo.max_depth)) / 2..(2_i32.pow(svo.max_depth)) / 2 {
            for k in -(2_i32.pow(svo.max_depth)) / 2..(2_i32.pow(svo.max_depth)) / 2 {
                if (i % 2 == 0 && j % 2 == 0 && k % 2 == 0) || ((i + 1) % 2 == 0 && (j + 1) % 2 == 0 && (k + 1) % 2 == 0) {
                    svo.insert(IVec3::new(i, j, k), 1);
                }
            }
        }
    }

    // svo.insert(IVec3::new(0, 0, 0), 1);
    // svo.insert(IVec3::new(1, 0, 0), 2);
    // svo.insert(IVec3::new(0, 1, 0), 2);
    // svo.insert(IVec3::new(0, 0, 1), 2);
    // svo.insert(IVec3::new(3, 0, -4), 2);

    // // Generate mesh for SVO
    // let camera_pos = camera_pos.translation;
    // let svo_mesh = svo.generate_mesh(camera_pos);
    // commands.spawn((
    //     Mesh3d(meshes.add(svo_mesh)),
    //     MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
    //     Transform::from_xyz(0.0, 0.0, 0.0),
    // ));

    let svo_octants_mesh = svo.generate_bounding_octants_mesh();
    commands.spawn((
        Mesh3d(meshes.add(svo_octants_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            unlit: true,
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    commands.insert_resource(svo);
}
