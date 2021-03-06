#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;

#[cfg(feature = "gl")]
extern crate gfx_backend_gl as back;

extern crate gfx_hal as hal;

extern crate glsl_to_spirv;
extern crate image;
extern crate winit;

extern crate learn_gfx;


use hal::format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags};
use hal::queue::Submission;
use hal::{
    buffer, command, format as f, image as i, memory as m, pass, pool, pso, window::Extent2D,
};
use hal::{Backbuffer, DescriptorPool, FrameSync, Primitive, SwapchainConfig};
use hal::{Device, Instance, PhysicalDevice, Surface, Swapchain};

use std::fs;
use std::io::{Cursor, Read};

#[cfg_attr(rustfmt, rustfmt_skip)]
const DIMS: Extent2D = Extent2D { width: 1024,height: 768 };

const ENTRY_NAME: &str = "main";

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
struct Vertex {
    a_Pos: [f32; 2],
    a_Uv: [f32; 2],
}

#[cfg_attr(rustfmt, rustfmt_skip)]
const QUAD: [Vertex; 6] = [
    Vertex { a_Pos: [ -0.5, 0.33 ], a_Uv: [0.0, 1.0] },
    Vertex { a_Pos: [  0.5, 0.33 ], a_Uv: [1.0, 1.0] },
    Vertex { a_Pos: [  0.5,-0.33 ], a_Uv: [1.0, 0.0] },

    Vertex { a_Pos: [ -0.5, 0.33 ], a_Uv: [0.0, 1.0] },
    Vertex { a_Pos: [  0.5,-0.33 ], a_Uv: [1.0, 0.0] },
    Vertex { a_Pos: [ -0.5,-0.33 ], a_Uv: [0.0, 0.0] },
];

const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

#[cfg(any(feature = "vulkan",feature = "dx12",feature = "gl"))]
fn main() {
    env_logger::init();

    let mut events_loop = winit::EventsLoop::new();

    let wb = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(
            DIMS.width as _,
            DIMS.height as _,
        )).with_title("quad".to_string());
    // instantiate backend
    #[cfg(not(feature = "gl"))]
    let (_window, _instance, mut adapters, mut surface) = {
        let window = wb.build(&events_loop).unwrap();
        let instance = back::Instance::create("learn gfx-rs quad", 1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();
        (window, instance, adapters, surface)
    };

    #[cfg(feature = "gl")]
        let (mut adapters, mut surface) = {
        let window = {
            let builder =
                back::config_context(back::glutin::ContextBuilder::new(), ColorFormat::SELF, None)
                    .with_vsync(true);
            back::glutin::GlWindow::new(wb, builder, &events_loop).unwrap()
        };
        let surface = back::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (adapters, surface)
    };

    adapters.iter().enumerate().for_each(|it|{
        println!("{} : {:?}\n", it.0,it.1.info);
    });

    let mut adapter = adapters.remove(0);
    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let limits = adapter.physical_device.limits();

    // Build a new device and associated command queues
    let (device, mut queue_group) = adapter
        .open_with::<_, hal::Graphics>(1, |family| surface.supports_queue_family(family))
        .unwrap();

    let mut command_pool = unsafe {
        device.create_command_pool_typed(&queue_group, pool::CommandPoolCreateFlags::empty())
    }.expect("Can't create command pool");

    // Setup renderpass and pipeline
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &[
                pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::SampledImage,
                    count: 1,
                    stage_flags: ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
                pso::DescriptorSetLayoutBinding {
                    binding: 1,
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                    stage_flags: ShaderStageFlags::FRAGMENT,
                    immutable_samplers: false,
                },
            ],
            &[],
        )
    }.expect("Can't create descriptor set layout");

    // Descriptors
    let mut desc_pool = unsafe {
        device.create_descriptor_pool(
            1, // sets
            &[
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::SampledImage,
                    count: 1,
                },
                pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Sampler,
                    count: 1,
                },
            ],
        )
    }.expect("Can't create descriptor pool");
    let desc_set = unsafe {
        desc_pool.allocate_set(&set_layout)
    }.unwrap();

    // Buffer allocations
    println!("Memory types: {:?}", memory_types);

    let buffer_stride = std::mem::size_of::<Vertex>() as u64;
    let buffer_len = QUAD.len() as u64 * buffer_stride;

    assert_ne!(buffer_len, 0);
    let mut vertex_buffer = unsafe {
        device.create_buffer(buffer_len, buffer::Usage::VERTEX)
    }.unwrap();

    let buffer_req = unsafe {
        device.get_buffer_requirements(&vertex_buffer)
    };

    let upload_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, mem_type)| {
            // type_mask is a bit field where each bit represents a memory type. If the bit is set
            // to 1 it means we can use that type for our buffer. So this code finds the first
            // memory type that has a `1` (or, is allowed), and is visible to the CPU.
            buffer_req.type_mask & (1 << id) != 0
                && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
        }).unwrap()
        .into();

    let buffer_memory = unsafe {
        device.allocate_memory(upload_type, buffer_req.size)
    }.unwrap();

    unsafe {
        device.bind_buffer_memory(&buffer_memory, 0, &mut vertex_buffer)
    }.unwrap();

    // TODO: check transitions: read/write mapping and vertex buffer read
    unsafe {
        let mut vertices = device
            .acquire_mapping_writer::<Vertex>(&buffer_memory, 0..buffer_req.size)
            .unwrap();
        vertices[0..QUAD.len()].copy_from_slice(&QUAD);
        device.release_mapping_writer(vertices).unwrap();
    }

    // Image
    let img_data = include_bytes!("../../data/quad/logo.png");

    let img = image::load(Cursor::new(&img_data[..]), image::PNG)
        .unwrap()
        .to_rgba();
    let (width, height) = img.dimensions();
    let kind = i::Kind::D2(width as i::Size, height as i::Size, 1, 1);
    let row_alignment_mask = limits.min_buffer_copy_pitch_alignment as u32 - 1;
    let image_stride = 4usize;
    let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
    let upload_size = (height * row_pitch) as u64;

    let mut image_upload_buffer = unsafe {
        device.create_buffer(upload_size, buffer::Usage::TRANSFER_SRC)
    }.unwrap();
    let image_mem_reqs = unsafe {
        device.get_buffer_requirements(&image_upload_buffer)
    };
    let image_upload_memory = unsafe {
        device.allocate_memory(upload_type, image_mem_reqs.size)
    }.unwrap();

    unsafe {
        device.bind_buffer_memory(&image_upload_memory, 0, &mut image_upload_buffer)
    }.unwrap();

    // copy image data into staging buffer
    unsafe {
        let mut data = device
            .acquire_mapping_writer::<u8>(&image_upload_memory, 0..image_mem_reqs.size)
            .unwrap();
        for y in 0..height as usize {
            let row = &(*img)
                [y * (width as usize) * image_stride..(y + 1) * (width as usize) * image_stride];
            let dest_base = y * row_pitch as usize;
            data[dest_base..dest_base + row.len()].copy_from_slice(row);
        }
        device.release_mapping_writer(data).unwrap();
    }

    let mut image_logo = unsafe {
        device
            .create_image(
                kind,
                1,
                ColorFormat::SELF,
                i::Tiling::Optimal,
                i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
                i::ViewCapabilities::empty(),
            )
    }.unwrap(); // TODO: usage
    let image_req = unsafe {
        device.get_image_requirements(&image_logo)
    };

    let device_type = memory_types
        .iter()
        .enumerate()
        .position(|(id, memory_type)| {
            image_req.type_mask & (1 << id) != 0
                && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
        }).unwrap()
        .into();
    let image_memory = unsafe {
        device.allocate_memory(device_type, image_req.size)
    }.unwrap();

    unsafe {
        device.bind_image_memory(&image_memory, 0, &mut image_logo)
    }.unwrap();
    let image_srv = unsafe {
        device
            .create_image_view(
                &image_logo,
                i::ViewKind::D2,
                ColorFormat::SELF,
                Swizzle::NO,
                COLOR_RANGE.clone(),
            )
    }.unwrap();

    let sampler = unsafe {
        device.create_sampler(i::SamplerInfo::new(i::Filter::Linear, i::WrapMode::Clamp))
    }.expect("Can't create sampler");

    unsafe {
        device.write_descriptor_sets(vec![
            pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 0,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Image(&image_srv, i::Layout::Undefined)),
            },
            pso::DescriptorSetWrite {
                set: &desc_set,
                binding: 1,
                array_offset: 0,
                descriptors: Some(pso::Descriptor::Sampler(&sampler)),
            },
        ]);
    }

    let mut frame_semaphore = device.create_semaphore().expect("Can't create semaphore");
    let mut frame_fence = device.create_fence(false).expect("Can't create fence"); // TODO: remove

    // copy buffer to texture
    unsafe {
        let mut cmd_buffer = command_pool.acquire_command_buffer::<command::OneShot>();
        cmd_buffer.begin();

        let image_barrier = m::Barrier::Image {
            states: (i::Access::empty(), i::Layout::Undefined)
                ..(i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal),
            target: &image_logo,
            families: None,
            range: COLOR_RANGE.clone(),
        };

        cmd_buffer.pipeline_barrier(
            PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
            m::Dependencies::empty(),
            &[image_barrier],
        );

        cmd_buffer.copy_buffer_to_image(
            &image_upload_buffer,
            &image_logo,
            i::Layout::TransferDstOptimal,
            &[command::BufferImageCopy {
                buffer_offset: 0,
                buffer_width: row_pitch / (image_stride as u32),
                buffer_height: height as u32,
                image_layers: i::SubresourceLayers {
                    aspects: f::Aspects::COLOR,
                    level: 0,
                    layers: 0..1,
                },
                image_offset: i::Offset { x: 0, y: 0, z: 0 },
                image_extent: i::Extent {
                    width,
                    height,
                    depth: 1,
                },
            }],
        );

        let image_barrier = m::Barrier::Image {
            states: (i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal)
                ..(i::Access::SHADER_READ, i::Layout::ShaderReadOnlyOptimal),
            target: &image_logo,
            families: None,
            range: COLOR_RANGE.clone(),
        };
        cmd_buffer.pipeline_barrier(
            PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
            m::Dependencies::empty(),
            &[image_barrier],
        );

        cmd_buffer.finish();

        queue_group.queues[0].submit_nosemaphores(Some(&cmd_buffer), Some(&mut frame_fence));

        device
            .wait_for_fence(&frame_fence, !0)
            .expect("Can't wait for fence");
    }

    let (caps, formats, _present_modes, _composite_alphas) =
        surface.compatibility(&mut adapter.physical_device);
    println!("formats: {:?}", formats);
    let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
        formats
            .iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .map(|format| *format)
            .unwrap_or(formats[0])
    });

    let swap_config = SwapchainConfig::from_caps(&caps, format, DIMS);
    println!("{:?}", swap_config);
    let extent = swap_config.extent.to_extent();

    let (mut swap_chain, mut backbuffer) = unsafe {
        device.create_swapchain(&mut surface, swap_config, None)
    }.expect("Can't create swapchain");

    let render_pass = {
        let attachment = pass::Attachment {
            format: Some(format),
            samples: 1,
            ops: pass::AttachmentOps::new(
                pass::AttachmentLoadOp::Clear,
                pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: pass::AttachmentOps::DONT_CARE,
            layouts: i::Layout::Undefined..i::Layout::Present,
        };

        let subpass = pass::SubpassDesc {
            colors: &[(0, i::Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        let dependency = pass::SubpassDependency {
            passes: pass::SubpassRef::External..pass::SubpassRef::Pass(0),
            stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses: i::Access::empty()
                ..(i::Access::COLOR_ATTACHMENT_READ | i::Access::COLOR_ATTACHMENT_WRITE),
        };

        unsafe {
            device.create_render_pass(&[attachment], &[subpass], &[dependency])
        }.expect("Can't create render pass")
    };
    let (mut frame_images, mut framebuffers) = match backbuffer {
        Backbuffer::Images(images) => {
            let pairs = images
                .into_iter()
                .map(|image| unsafe {
                    let rtv = device
                        .create_image_view(
                            &image,
                            i::ViewKind::D2,
                            format,
                            Swizzle::NO,
                            COLOR_RANGE.clone(),
                        ).unwrap();
                    (image, rtv)
                }).collect::<Vec<_>>();
            let fbos = pairs
                .iter()
                .map(|&(_, ref rtv)| unsafe {
                    device
                        .create_framebuffer(&render_pass, Some(rtv), extent)
                        .unwrap()
                }).collect();
            (pairs, fbos)
        }
        Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
    };

    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            std::iter::once(&set_layout),
            &[(pso::ShaderStageFlags::VERTEX, 0..8)],
        )
    }.expect("Can't create pipeline layout");
    let pipeline = {
        let vs_module = {
            let glsl = fs::read_to_string("data/quad/quad.vert").unwrap();
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Vertex)
                .unwrap()
                .bytes()
                .map(|b| b.unwrap())
                .collect();
            unsafe {
                device.create_shader_module(&spirv)
            }.unwrap()
        };
        let fs_module = {
            let glsl = fs::read_to_string("data/quad/quad.frag").unwrap();
            let spirv: Vec<u8> = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Fragment)
                .unwrap()
                .bytes()
                .map(|b| b.unwrap())
                .collect();
            unsafe {
                device.create_shader_module(&spirv)
            }.unwrap()
        };

        let pipeline = {
            let (vs_entry, fs_entry) = (
                pso::EntryPoint {
                    entry: ENTRY_NAME,
                    module: &vs_module,
                    specialization: pso::Specialization {
                        constants: &[pso::SpecializationConstant { id: 0, range: 0..4 }],
                        data: unsafe { std::mem::transmute::<&f32, &[u8; 4]>(&0.8f32) },
                    },
                },
                pso::EntryPoint {
                    entry: ENTRY_NAME,
                    module: &fs_module,
                    specialization: pso::Specialization::default(),
                },
            );

            let shader_entries = pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let subpass = Subpass {
                index: 0,
                main_pass: &render_pass,
            };

            let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                pso::Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );
            pipeline_desc.blender.targets.push(pso::ColorBlendDesc(
                pso::ColorMask::ALL,
                pso::BlendState::ALPHA,
            ));
            pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                binding: 0,
                stride: std::mem::size_of::<Vertex>() as u32,
                rate: 0,
            });

            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 0,
                binding: 0,
                element: pso::Element {
                    format: f::Format::Rg32Float,
                    offset: 0,
                },
            });
            pipeline_desc.attributes.push(pso::AttributeDesc {
                location: 1,
                binding: 0,
                element: pso::Element {
                    format: f::Format::Rg32Float,
                    offset: 8,
                },
            });

            unsafe {
                device.create_graphics_pipeline(&pipeline_desc, None)
            }
        };

        unsafe {
            device.destroy_shader_module(vs_module);
        }
        unsafe {
            device.destroy_shader_module(fs_module);
        }

        pipeline.unwrap()
    };

    // Rendering setup
    let mut viewport = pso::Viewport {
        rect: pso::Rect {
            x: 0,
            y: 0,
            w: extent.width as _,
            h: extent.height as _,
        },
        depth: 0.0..1.0,
    };

    //
    let mut running = true;
    let mut recreate_swapchain = false;
    let mut resize_dims = Extent2D {
        width: 0,
        height: 0,
    };
    while running {
        events_loop.poll_events(|event| {
            if let winit::Event::WindowEvent { event, .. } = event {
                #[allow(unused_variables)]
                    match event {
                    winit::WindowEvent::KeyboardInput {
                        input:
                        winit::KeyboardInput {
                            virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    }
                    | winit::WindowEvent::CloseRequested => running = false,
                    winit::WindowEvent::Resized(dims) => {
                        println!("resized to {:?}", dims);

                        recreate_swapchain = true;
                        resize_dims.width = dims.width as u32;
                        resize_dims.height = dims.height as u32;
                    }
                    _ => (),
                }
            }
        });

        // Window was resized so we must recreate swapchain and framebuffers
        if recreate_swapchain {
            device.wait_idle().unwrap();

            let (caps, formats, _present_modes, _composite_alphas) =
                surface.compatibility(&mut adapter.physical_device);
            // Verify that previous format still exists so we may reuse it.
            assert!(formats.iter().any(|fs| fs.contains(&format)));

            let swap_config = SwapchainConfig::from_caps(&caps, format, resize_dims);
            println!("{:?}", swap_config);
            let extent = swap_config.extent.to_extent();

            let (new_swap_chain, new_backbuffer) = unsafe {
                device.create_swapchain(&mut surface, swap_config, Some(swap_chain))
            }.expect("Can't create swapchain");

            unsafe {
                // Clean up the old framebuffers, images and swapchain
                for framebuffer in framebuffers {
                    device.destroy_framebuffer(framebuffer);
                }
                for (_, rtv) in frame_images {
                    device.destroy_image_view(rtv);
                }
            }

            backbuffer = new_backbuffer;
            swap_chain = new_swap_chain;

            let (new_frame_images, new_framebuffers) = match backbuffer {
                Backbuffer::Images(images) => {
                    let pairs = images
                        .into_iter()
                        .map(|image| unsafe {
                            let rtv = device.create_image_view(
                                &image,
                                i::ViewKind::D2,
                                format,
                                Swizzle::NO,
                                COLOR_RANGE.clone(),
                            ).unwrap();
                            (image, rtv)
                        }).collect::<Vec<_>>();
                    let fbos = pairs
                        .iter()
                        .map(|&(_, ref rtv)| unsafe {
                            device
                                .create_framebuffer(&render_pass, Some(rtv), extent)
                                .unwrap()
                        }).collect();
                    (pairs, fbos)
                }
                Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
            };

            framebuffers = new_framebuffers;
            frame_images = new_frame_images;
            viewport.rect.w = extent.width as _;
            viewport.rect.h = extent.height as _;
            recreate_swapchain = false;
        }

        let frame: hal::SwapImageIndex = unsafe {
            device.reset_fence(&frame_fence).unwrap();
            command_pool.reset();
            match swap_chain.acquire_image(!0, FrameSync::Semaphore(&mut frame_semaphore)) {
                Ok(i) => i,
                Err(_) => {
                    recreate_swapchain = true;
                    continue;
                }
            }
        };

        // Rendering
        let mut cmd_buffer = command_pool.acquire_command_buffer::<command::OneShot>();
        unsafe {
            cmd_buffer.begin();

            cmd_buffer.set_viewports(0, &[viewport.clone()]);
            cmd_buffer.set_scissors(0, &[viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&pipeline);
            cmd_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));
            cmd_buffer.bind_graphics_descriptor_sets(&pipeline_layout, 0, Some(&desc_set), &[]); //TODO

            {
                let mut encoder = cmd_buffer.begin_render_pass_inline(
                    &render_pass,
                    &framebuffers[frame as usize],
                    viewport.rect,
                    &[command::ClearValue::Color(command::ClearColor::Float([
                        0.8, 0.8, 0.8, 1.0,
                    ]))],
                );
                encoder.draw(0..6, 0..1);
            }

            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: Some((&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)),
                signal_semaphores: &[],
            };
            queue_group.queues[0].submit(submission, Some(&mut frame_fence));

            // TODO: replace with semaphore
            device.wait_for_fence(&frame_fence, !0).unwrap();
            command_pool.free(Some(cmd_buffer));

            // present frame
            if let Err(_) = swap_chain.present_nosemaphores(&mut queue_group.queues[0], frame) {
                recreate_swapchain = true;
            }
        }
    }

    // cleanup!
    device.wait_idle().unwrap();
    unsafe {
        device.destroy_command_pool(command_pool.into_raw());
        device.destroy_descriptor_pool(desc_pool);
        device.destroy_descriptor_set_layout(set_layout);

        device.destroy_buffer(vertex_buffer);
        device.destroy_buffer(image_upload_buffer);
        device.destroy_image(image_logo);
        device.destroy_image_view(image_srv);
        device.destroy_sampler(sampler);
        device.destroy_fence(frame_fence);
        device.destroy_semaphore(frame_semaphore);
        device.destroy_render_pass(render_pass);
        device.free_memory(buffer_memory);
        device.free_memory(image_memory);
        device.free_memory(image_upload_memory);
        device.destroy_graphics_pipeline(pipeline);
        device.destroy_pipeline_layout(pipeline_layout);
        for framebuffer in framebuffers {
            device.destroy_framebuffer(framebuffer);
        }
        for (_, rtv) in frame_images {
            device.destroy_image_view(rtv);
        }

        device.destroy_swapchain(swap_chain);
    }
}


#[cfg(not(any(feature = "vulkan",feature = "dx12",feature = "gl")))]
fn main() {
    println!("features must be one of vulkan dx12 gl!");
}


