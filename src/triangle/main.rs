#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as bankend;
extern crate gfx_hal as hal;

extern crate winit;
extern crate image;
extern crate glsl_to_spirv;

extern crate learn_gfx;

use std::fs::File;
use std::io::{Write,Read};
use winit::WindowEvent;

use hal::{  Instance,
            Device,
            PhysicalDevice,
            pool,
            Surface,
            Primitive,
            SwapchainConfig,
            Swapchain,
            FrameSync
};
use hal::pso::{
    PipelineStage,
    Stage,
    EntryPoint,
    GraphicsShaderSet,
    GraphicsPipelineDesc,
    Rasterizer,
    ColorBlendDesc,
    ColorMask,
    BlendState,
    Viewport, Rect
};


use hal::pass::{
    Attachment,
    AttachmentOps,
    AttachmentLoadOp,
    AttachmentStoreOp,
    SubpassDesc,
    AttachmentLayout,
    SubpassDependency,
    SubpassRef,
    Subpass
};

use hal::format::{
    Format,
    Aspects,
    Swizzle,
    ChannelType
};

use hal::image::{
    Layout,
    Access,
    SubresourceRange,
    ViewKind
};

use hal::window::{
    Extent2D,
    Backbuffer
};
use hal::command::{ ClearValue,ClearColor,RenderPassInlineEncoder};
use hal::queue::Submission;

use glsl_to_spirv::compile;
use glsl_to_spirv::ShaderType;

use learn_gfx::comm::pick_adapter;


const W:u32 = 800;
const H:u32 = 600;
const TITLE:&'static str = "triangle";

const VERTEX_SHADER_PATH:&'static str = "data/triangle/triangle.vert";
const FRAG_SHADER_PATH:&'static str = "data/triangle/triangle.frag";

//參考 https://github.com/Mistodon/gfx-hal-tutorials/blob/master/src/bin/part00-triangle.rs
#[cfg(feature = "vulkan")]
fn main()
{
    let mut events_loop = winit::EventsLoop::new();
    let wb = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new( W as _,H as _ ))
        .with_title(TITLE);

    let (_window,_instance,mut surface,adapters) = {
        let window = wb.build(&events_loop).unwrap();
        let instance = bankend::Instance::create(TITLE,1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();

        (window,instance,surface,adapters)
    };

    adapters.iter().enumerate().for_each(|it|{
        println!("{} {:?}",it.0,it.1.info);
    });

    let res = pick_adapter(adapters,&surface);
    let (device,mut queue_group,adapter) = if let Ok(adapter) = res{
        adapter
    }else {
        panic!("can not pick a adapter!");
    };

    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let limits = adapter.physical_device.limits();

    let mut command_pool = unsafe { device.create_command_pool_typed(&queue_group,pool::CommandPoolCreateFlags::empty()) }.unwrap();

    let physical_device = &(adapter.physical_device);

    let (caps,formats,..) = surface.compatibility(physical_device);

    println!("choose adapter = {:?}",adapter.info);
    println!("caps = {:?}",caps);
    println!("formats = {:?}",formats);

    let format = if let Some(fs) = formats{
        fs.into_iter().find(|it|{
            it.base_format().1 == ChannelType::Srgb
        }).unwrap()
    }else{
        Format::Rgba8Srgb
    };

    let render_pass = {
        let color_attachment = Attachment {
            format: Some(format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::Present
        };

        let sub_pass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[]
        };

        let dependency = SubpassDependency {
            passes : SubpassRef::External..SubpassRef::Pass(0),
            stages : PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
            accesses : Access::empty() .. (Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE)
        };

        unsafe { device.create_render_pass(&[color_attachment],&[sub_pass],&[dependency]).unwrap() }
    };

    let pipeline_layout = unsafe {device.create_pipeline_layout(&[],&[]).unwrap() };

    let vertex_shader_module = {
        let mut content_str = String::new();
        File::open(VERTEX_SHADER_PATH).unwrap().read_to_string(&mut content_str);
        let spirv: Vec<u8> = glsl_to_spirv::compile(&content_str, glsl_to_spirv::ShaderType::Vertex)
            .unwrap()
            .bytes()
            .map(|b| b.unwrap())
            .collect();
        unsafe {
            device.create_shader_module(&spirv)
        }.unwrap()
    };

    let fragment_shader_module = {
        let mut content_str = String::new();
        File::open(FRAG_SHADER_PATH).unwrap().read_to_string(&mut content_str);
        let spirv: Vec<u8> = glsl_to_spirv::compile(&content_str, glsl_to_spirv::ShaderType::Fragment)
            .unwrap()
            .bytes()
            .map(|b| b.unwrap())
            .collect();
        unsafe {
            device.create_shader_module(&spirv)
        }.unwrap()
    };

    let pipeline = {
        let vs_entry = EntryPoint::<bankend::Backend>{
            entry  : "main",
            module : &vertex_shader_module,
            specialization : Default::default()
        };

        let fs_entry = EntryPoint::<bankend::Backend>{
            entry  : "main",
            module : &fragment_shader_module,
            specialization : Default::default()
        };

        let shader_set = GraphicsShaderSet::<bankend::Backend>{
            vertex : vs_entry,
            hull: None,
            domain :None,
            geometry : None,
            fragment : Some(fs_entry)
        };

        let sub_pass = Subpass::<bankend::Backend>{
            index : 0,
            main_pass : &render_pass
        };

        let mut pipeline_desc = GraphicsPipelineDesc::new(
            shader_set,
            Primitive::TriangleList,
            Rasterizer::FILL,
            &pipeline_layout,
            sub_pass
        );

        pipeline_desc.blender.targets.push(ColorBlendDesc(ColorMask::ALL,BlendState::ALPHA));

        unsafe { device.create_graphics_pipeline(&pipeline_desc,None).unwrap() }
    };

    let swapchain_config = SwapchainConfig::from_caps(&caps,format,
                                                      Extent2D{ width:W,height:H });
    let extent = swapchain_config.extent.to_extent();

    let (mut swapchain, backbuffer) = unsafe { device.create_swapchain(&mut surface,swapchain_config,None).unwrap()};

    let (image_views,framebuffers) = match backbuffer{
        Backbuffer::Images(images) => {
            let color_range = SubresourceRange{
                aspects : Aspects::COLOR,
                levels : 0..1,
                layers : 0..1
            };
            let image_views = images.iter().map(|it|{
                unsafe { device.create_image_view(
                   it,
                   ViewKind::D2,
                   format,
                   Swizzle::NO,
                   color_range.clone()
               ).unwrap() }
            }).collect::<Vec<_>>();

            let fbos = image_views.iter().map(|it|{
                unsafe {device.create_framebuffer(&render_pass,vec![it],extent).unwrap()}
            }).collect::<Vec<_>>();

            (image_views,fbos)
        },
        Backbuffer::Framebuffer(framebuffer) => {
            (vec![],vec![framebuffer])
        }
    };

    let frame_semaphore = device.create_semaphore().unwrap();
    let present_semaphore = device.create_semaphore().unwrap();

    loop{
        let mut quiting = false;

        events_loop.poll_events(|we|{
            if let winit::Event::WindowEvent {event,..} = we{
                match event {
                    WindowEvent::CloseRequested => { quiting = true;},
                    _ => {}
                }
            }
        });

        if quiting { break; }
        let frame_index = unsafe {
            command_pool.reset();
            swapchain.acquire_image(!0, FrameSync::Semaphore(&frame_semaphore))
                .expect("Failed to acquire frame!")
        };
        let finished_command_buffer = {
            let mut command_buffer = command_pool.acquire_command_buffer::<hal::command::OneShot>();

            let viewport = Viewport{
                rect : Rect { x : 0, y: 0, w : extent.width as _, h : extent.height as _  },
                depth : 0.0 .. 1.0
            };

            let view_port_rect = viewport.rect;

            unsafe {
                command_buffer.set_scissors(0, &[view_port_rect]);
                command_buffer.set_viewports(0, &[viewport]);
            }


            unsafe{
                command_buffer.bind_graphics_pipeline(&pipeline);
                let mut encode: RenderPassInlineEncoder<_> = command_buffer.begin_render_pass_inline(
                    &render_pass,
                    &framebuffers[frame_index as usize],
                    view_port_rect,
                    &[ClearValue::Color(ClearColor::Float([0.0, 0.0, 0.0, 1.0]))]
                );

                encode.draw(0..3, 0..1);
            }
            unsafe { command_buffer.finish();}
            command_buffer
        };
        let submission = Submission{
            command_buffers: Some(&finished_command_buffer),
            wait_semaphores: Some((&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)),
            signal_semaphores : vec![&present_semaphore],
        };
//        let submission = Submission::new()
//            .wait_on(&[(&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
//            .signal(&[])
//            .submit(vec![finished_command_buffer]);

        unsafe {
            queue_group.queues[0].submit(submission, None);

            swapchain
            .present(
                &mut queue_group.queues[0],
                frame_index,
                vec![&present_semaphore],
            ).expect("Present failed");
        }
    }

    device.wait_idle().unwrap();
    unsafe {
        device.destroy_command_pool(command_pool.into_raw());
        device.destroy_render_pass(render_pass);

        device.destroy_graphics_pipeline(pipeline);
        device.destroy_pipeline_layout(pipeline_layout);

        for framebuffer in framebuffers {
            device.destroy_framebuffer(framebuffer);
        }

        for image_view in image_views {
            device.destroy_image_view(image_view);
        }
        device.destroy_swapchain(swapchain);

        device.destroy_shader_module(vertex_shader_module);
        device.destroy_shader_module(fragment_shader_module);

        device.destroy_semaphore(frame_semaphore);
        device.destroy_semaphore(present_semaphore);
    }
}



#[cfg(not(feature = "vulkan"))]
fn main()
{
    println!("feature must be vulkan!");
}