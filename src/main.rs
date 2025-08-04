use bevy::{
    prelude::*,
    window::{PresentMode, WindowResolution},
};

use crate::octree::setup_voxel_world;

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
        .add_systems(Startup, (setup, setup_voxel_world))
        .run();
}

/// set up a simple 3D scene
fn setup(mut commands: Commands, // , mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>
) {
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
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10., 30.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
