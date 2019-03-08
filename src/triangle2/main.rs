#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as backend;

#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as backend;

#[cfg(feature = "gl")]
extern crate gfx_backend_gl as backend;

extern crate gfx_hal as hal;
extern crate winit;
extern crate learn_gfx;
extern crate cgmath;
#[cfg(any(feature = "vulkan",feature = "dx12",feature = "gl"))]
type TOfB = backend::Backend;


use hal::{ Instance,PhysicalDevice,Device,Surface,SurfaceCapabilities,SwapchainConfig,memory,CommandPool,command,Submission,QueueGroup,Primitive,Swapchain};
use hal::window::{ Extent2D,Backbuffer };
use hal::image::{ViewKind,Extent,SubresourceRange,Layout,Access,Kind,Tiling,Usage,ViewCapabilities};
use hal::buffer;
use hal::pass::{Attachment,
                AttachmentOps,
                AttachmentLoadOp,
                AttachmentStoreOp,
                SubpassDesc,
                SubpassDependency,
                SubpassRef,
                Subpass
};
use hal::command::ClearDepthStencil;
use hal::pso::*;
use hal::format::{Format,ChannelType,Swizzle,Aspects,ImageFeature};
use hal::pool::{CommandPoolCreateFlags};
use hal::adapter::{ MemoryType };
use learn_gfx::comm::pick_adapter;
use std::mem::size_of;
use std::ptr::{copy};
use cgmath::{Matrix4, Vector3, Deg, Rad};
use winit::dpi::LogicalPosition;

const W:u32 = 800;
const H:u32 = 600;
const TITLE:&'static str = "triangle2";

//shader data
const VERTEX_SHADER_DATA :&[u8] = include_bytes!("../../data/triangle2/triangle.vert.spv");
const FRAGMENT_SHADER_DATA :&[u8] = include_bytes!("../../data/triangle2/triangle.frag.spv");

type Mat4 = Matrix4<f32>;
type Vec3 = Vector3<f32>;

#[repr(C)]
struct Ubo{
    projection : Mat4,
    model : Mat4,
    view : Mat4
}

struct Triangle{
    pos : Vec3,
    rotate : Vec3
}

struct DepthStencil<B : hal::Backend>{
    pub image : B::Image,
    pub memory : B::Memory,
    pub view : B::ImageView
}

impl<B : hal::Backend> DepthStencil<B>
{
    pub fn destroy(self,device:&B::Device)
    {
        unsafe {
            device.destroy_image(self.image);
            device.free_memory(self.memory);
            device.destroy_image_view(self.view);
        }
    }
}

#[cfg(any(feature = "vulkan",feature = "dx12",feature = "gl"))]
fn main()
{
    let mut events_loop = winit::EventsLoop::new();
    let wb = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(W as f64,H as f64))
        .with_title(TITLE);

    #[cfg(not(feature = "gl"))]
    let (_window,_instance,mut surface, adapters) = {
        let window = wb.build(&events_loop).unwrap();
        let instance = backend::Instance::create(TITLE,1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();

        (window,instance,surface,adapters)
    };

    #[cfg(feature = "gl")]
        let ( adapters, mut surface) = {
        let window = {
            let builder =
                backend::config_context(backend::glutin::ContextBuilder::new(), Format::Rgba8Srgb, None)
                    .with_vsync(true);
            backend::glutin::GlWindow::new(wb, builder, &events_loop).unwrap()
        };
        let surface = backend::Surface::from_window(window);
        let adapters = surface.enumerate_adapters();
        (adapters, surface)
    };

    let (device,mut queue_group,adapter) = if let Ok(res) = pick_adapter(adapters,&surface)
    {
        res
    }else {
        panic!("failed to pick a adapter!")
    };

    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let _limits = adapter.physical_device.limits();
    let mut command_pool = unsafe {
        device.create_command_pool_typed(&queue_group,CommandPoolCreateFlags::empty())
    }.unwrap();

    let physical_device = &adapter.physical_device;

    let (caps,formats,..) = surface.compatibility(physical_device);

    #[cfg(not(feature = "gl"))]
    let depth_format = if let Some(f) = get_depth_format::<TOfB>(physical_device){
        f
    }else{
        panic!("Not get depth format!");
    };

    #[cfg(feature = "gl")]
    let depth_format = Format::Rgba8Srgb;

    let mut depth_stencil = unsafe{ create_depth_stencil::<TOfB>(&device,depth_format,W,H,&memory_types) };
    println!("choose adapter = {:?}",adapter.info);
    println!("caps = {:?}",caps);
    println!("formats = {:?}",formats);

    let format = if let Some(fs) = formats{
        fs.into_iter().find(|it|{
            it.base_format().1  == ChannelType::Srgb
        }).unwrap()
    }else{
        Format::Rgba8Srgb
    };

    let render_pass = create_render_pass::<TOfB>(&device,format,depth_format).ok().unwrap();
    let (mut swap_chain,mut _extent,mut image_views,mut frame_buffers) = create_swapchain::<TOfB>(&device,
                                                                             &mut surface,
                                                                             &render_pass,
                                                                             &caps,format,
                                                                             W ,H,None,None,None,&depth_stencil);
    let (vertex_buffer,vertex_mem) = unsafe{ create_vertex_buffer(&device,&memory_types,&mut command_pool,&mut queue_group).unwrap() };

    let (index_buffer,index_mem) = unsafe{ create_index_buffer(&device,&memory_types,&mut command_pool,&mut queue_group).unwrap() };

    let (uniform_buffer,uniform_mem) = unsafe{ create_buffer::<TOfB>(size_of::<Ubo>() as u64,buffer::Usage::UNIFORM,&device,&memory_types).unwrap() };

    let mut triangl = Triangle{
        pos : Vector3{ x:0.0f32,y:0.0f32,z:-4.0f32 },
        rotate : Vector3{ x : 0.0,y: 0.0,z : 0.0}
    };

    let mut width = W;
    let mut height = H;

    let mut view_rot = Vector3::<f32>{x:0.0,y:0.0,z:0.0};

    let mut descriptor_pool = create_descriptor_pool::<TOfB>(&device).unwrap();
    let descriptor_set_layout = create_descriptor_set_layout::<TOfB>(&device).unwrap();
    let descriptor_set = unsafe{ descriptor_pool.allocate_set(&descriptor_set_layout).unwrap() };

    let write_descriptor_set = DescriptorSetWrite{
        set : &descriptor_set,
        binding : 0,
        array_offset : 0,
        descriptors : &[ Descriptor::Buffer(&uniform_buffer,None..None)]
    };
    unsafe { device.write_descriptor_sets(Some(write_descriptor_set)); }
    update_uniform_buffer::<TOfB>(&device,&uniform_mem,&triangl,view_rot,W as f32 / H as f32);
    let pipeline_layout = unsafe { device.create_pipeline_layout(
        vec![&descriptor_set_layout],
        &[]).unwrap() };

    let pipeline = create_pipeline::<TOfB>(&device,&pipeline_layout,&render_pass).unwrap();
    let render_semaphore = device.create_semaphore().unwrap();
    let present_semaphore = device.create_semaphore().unwrap();
    let mut frame_fence = device.create_fence(true).unwrap();

    let mut running = true;
    let mut recreate_swapchain = false;
    let mut left_button_down = false;
    let mut cursor_pos = LogicalPosition::new(0.0,0.0);

    while running {
        events_loop.poll_events(|event|{
            if let winit::Event::WindowEvent{event,..} = event{
                match event {
                    winit::WindowEvent::KeyboardInput { input : winit::KeyboardInput{ virtual_keycode: Some(winit::VirtualKeyCode::Escape),.. },.. } |
                    winit::WindowEvent::CloseRequested  =>  running = false,

                    winit::WindowEvent::Resized(dims) => {
                        width = dims.width as _;
                        height = dims.height as _;
                        recreate_swapchain = true;
                    },
                    winit::WindowEvent::MouseInput {button:winit::MouseButton::Left ,state,..} => {
                        match state {
                            winit::ElementState::Pressed => { left_button_down = true; },
                            winit::ElementState::Released => { left_button_down = false; }
                        }
                    },
                    winit::WindowEvent::CursorMoved { position,.. } => {
                        if left_button_down{
                            let offset = LogicalPosition::new(position.x - cursor_pos.x,position.y - cursor_pos.y);
                            triangl.rotate.y += (offset.x * 1.25) as f32;
//                            triangl.rotate.x -= (offset.y * 1.25) as f32;
//                            view_rot.y += (offset.x ) as f32;
                            view_rot.x -= (offset.y * 1.25) as f32;
                        }
                        cursor_pos = position;
                    },
                    _ => {}
                }
            }
        });

        if recreate_swapchain{
            device.wait_idle().unwrap();
            let (caps_, ..) =
                surface.compatibility(physical_device);
            depth_stencil.destroy(&device);
            depth_stencil = unsafe{ create_depth_stencil::<TOfB>(&device,depth_format,width,height,&memory_types) };
            let (swapchain_,
                extent_,
                image_views_,
                framebuffers_)  = create_swapchain::<TOfB>(&device,&mut surface, &render_pass, &caps_,format,
                                                                       width,height,Some(swap_chain),Some(image_views),
                                                                       Some(frame_buffers),
                                                                        &depth_stencil);

            swap_chain = swapchain_;
            _extent = extent_;
            image_views = image_views_;
            frame_buffers = framebuffers_;

            recreate_swapchain = false;
        }
        if !running { break; }

        let frame_index = unsafe {
            device.reset_fence(&frame_fence).unwrap();
            command_pool.reset();
            match swap_chain.acquire_image(!0, hal::FrameSync::Semaphore(&render_semaphore)) {
                Ok(i) => i,
                Err(_) => {
                    recreate_swapchain = true;
                    continue;
                }
            }
        };

        unsafe {
            let mut draw_buffer = command_pool.acquire_command_buffer::<command::OneShot>();
            draw_buffer.begin();

            let viewport = Viewport{
                rect : Rect{ x:0,y:0,w:width as _,h:height as _ },
                depth : 0.0f32..1.0f32
            };

            let index_buffer_view = buffer::IndexBufferView{
                buffer : &index_buffer,
                offset : 0,
                index_type : hal::IndexType::U32
            };

            draw_buffer.set_viewports(0,&[viewport.clone()]);
            draw_buffer.set_scissors(0,&[viewport.rect]);
            draw_buffer.bind_graphics_pipeline(&pipeline);
            draw_buffer.bind_vertex_buffers(0, Some((&vertex_buffer, 0)));
            draw_buffer.bind_index_buffer(index_buffer_view);
            draw_buffer.bind_graphics_descriptor_sets(&pipeline_layout,0,Some(&descriptor_set),&[]);

            {
                let mut encoder = draw_buffer.begin_render_pass_inline(
                    &render_pass,
                    &frame_buffers[frame_index as usize],
                    viewport.rect,
                    &[command::ClearValue::Color(command::ClearColor::Float([0.0, 0.0, 0.0, 1.0] ) ),
                        command::ClearValue::DepthStencil(ClearDepthStencil(1.0,0))],
                );
                encoder.draw_indexed(0..3,0,0..1);
            }
            draw_buffer.finish();

            let submission = Submission{
                command_buffers: Some(&draw_buffer),
                wait_semaphores: vec![(&render_semaphore, PipelineStage::COLOR_ATTACHMENT_OUTPUT)],
                signal_semaphores : vec![&present_semaphore],
            };



            queue_group.queues[0].submit(submission, Some(&mut frame_fence));

            device.wait_for_fence(&frame_fence, !0).unwrap();

            command_pool.free(Some(draw_buffer));

            if let Err(_) = swap_chain.present(
                &mut queue_group.queues[0],
                frame_index,
                vec![&present_semaphore],
            ){
                recreate_swapchain = true;
            }

        }
        update_uniform_buffer::<TOfB>(&device,&uniform_mem,&triangl,view_rot,width as f32 / height as f32);
    }

    unsafe {
        device.wait_idle().unwrap();
        device.destroy_semaphore(render_semaphore);
        device.destroy_semaphore(present_semaphore);
        device.destroy_fence(frame_fence);

        device.destroy_swapchain(swap_chain);
        device.destroy_command_pool(command_pool.into_raw());
        device.destroy_render_pass(render_pass);
        device.free_memory(vertex_mem);
        device.destroy_buffer(vertex_buffer);
        device.free_memory(index_mem);
        device.destroy_buffer(index_buffer);
        device.free_memory(uniform_mem);
        device.destroy_buffer(uniform_buffer);
        device.destroy_graphics_pipeline(pipeline);
        device.destroy_pipeline_layout(pipeline_layout);
        descriptor_pool.free_sets(vec![descriptor_set]);
        device.destroy_descriptor_pool(descriptor_pool);
        device.destroy_descriptor_set_layout(descriptor_set_layout);
        depth_stencil.destroy(&device);
        for iv in image_views{
            device.destroy_image_view(iv);
        }
        for fb in frame_buffers{
            device.destroy_framebuffer(fb);
        }
    }
}

fn create_render_pass<B : hal::Backend>(device:&B::Device,format:Format,depth_format:Format) -> Result<B::RenderPass, hal::device::OutOfMemory>
{
    let color_attachment = Attachment{
        format : Some(format),
        samples: 1,
        ops: AttachmentOps::new(AttachmentLoadOp::Clear,AttachmentStoreOp::Store),
        stencil_ops : AttachmentOps::DONT_CARE,
        layouts:  Layout::Undefined..Layout::Present
    };

    let depth_attachment = Attachment{
        format : Some(depth_format),
        samples : 1,
        ops : AttachmentOps::new(AttachmentLoadOp::Clear,AttachmentStoreOp::DontCare),
        stencil_ops : AttachmentOps::DONT_CARE,
        layouts : Layout::Undefined..Layout::DepthStencilAttachmentOptimal
    };
    let sub_pass_desc = SubpassDesc{
        colors : &[(0,Layout::ColorAttachmentOptimal)],
        depth_stencil : Some(&(1,Layout::DepthStencilAttachmentOptimal)),
        inputs : &[],
        resolves : &[],
        preserves : &[]
    };
    let sub_pass_dependency = SubpassDependency{
        passes : SubpassRef::External .. SubpassRef::Pass(0),
        stages : PipelineStage::BOTTOM_OF_PIPE ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
        accesses: Access::MEMORY_READ..(Access::COLOR_ATTACHMENT_WRITE | Access::COLOR_ATTACHMENT_READ)
    };

    unsafe { device.create_render_pass(&[color_attachment,depth_attachment],
                                       &[sub_pass_desc],
                                       &[sub_pass_dependency]) }
}

fn create_swapchain<B:hal::Backend>(device:&B::Device,
                                    surface:& mut B::Surface,
                                    render_pass:&B::RenderPass,
                                    caps:&SurfaceCapabilities,
                                    format:Format,w:u32,h:u32,
                                    old_swapchain:Option<B::Swapchain>,
                                    old_ivs : Option<Vec<B::ImageView>>,
                                    old_fbs : Option<Vec<B::Framebuffer>>,
                                    depth_stencil : &DepthStencil<B>) -> (B::Swapchain,Extent,Vec<B::ImageView>,Vec<B::Framebuffer>)
{
    let swapchain_config = SwapchainConfig::from_caps(caps,format,
                                                      Extent2D{ width:w,height:h });
    let extent = swapchain_config.extent.to_extent();

    let ( swapchain, backbuffer) = unsafe { device.create_swapchain(surface,swapchain_config,old_swapchain).unwrap()};

    if let Some(ivs) = old_ivs{
        for iv in ivs{
            unsafe{ device.destroy_image_view(iv); }
        }
    }

    if let Some(fbs) = old_fbs{
        for fb in fbs{
            unsafe{ device.destroy_framebuffer(fb); }
        }
    }

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
                unsafe {device.create_framebuffer(render_pass,vec![it,&(depth_stencil.view)],extent).unwrap()}
            }).collect::<Vec<_>>();

            (image_views,fbos)
        },
        Backbuffer::Framebuffer(framebuffer) => {
            (vec![],vec![framebuffer])
        }
    };
    (swapchain,extent,image_views,framebuffers)
}

#[repr(C)]
struct Vertex{
    pos : [f32;3],
    color : [f32;3]
}

unsafe fn create_vertex_buffer<B:hal::Backend>(device: &B::Device,
                                               mem_types:&Vec<MemoryType>,
                                               comm_pool :&mut CommandPool<B,hal::Graphics>,
                                               queue_group:&mut QueueGroup<B,hal::Graphics>) -> Option<(B::Buffer,B::Memory)>
{
    let vertices = vec![
        Vertex{ pos:[  1.0f32,  1.0f32, 0.0f32 ],   color:[ 1.0f32, 0.0f32, 0.0f32  ] },
        Vertex{ pos:[  -1.0f32,  1.0f32, 0.0f32 ] , color:[ 0.0f32, 1.0f32, 0.0f32  ] },
        Vertex{ pos:[  0.0f32, -1.0f32, 0.0f32 ] , color:[ 0.0f32, 0.0f32, 1.0f32  ] }
    ];

    let vertex_byte_size = size_of::<Vertex>() * 3;

    copy_buffer_stage(&(vertices[0]) as *const Vertex as _,vertex_byte_size,buffer::Usage::VERTEX,device,mem_types,comm_pool,queue_group)

}


unsafe fn create_index_buffer<B:hal::Backend>(device: &B::Device,
                                               mem_types:&Vec<MemoryType>,
                                               comm_pool :&mut CommandPool<B,hal::Graphics>,
                                               queue_group:&mut QueueGroup<B,hal::Graphics>) -> Option<(B::Buffer,B::Memory)>
{
    let indices:Vec<u32> = vec![
        0,1,2
    ];

    let byte_size = size_of::<u32>() * 3;

    copy_buffer_stage(&(indices[0]) as *const u32 as _,byte_size  ,buffer::Usage::INDEX,device,mem_types,comm_pool,queue_group)
}

unsafe fn copy_buffer_stage<B:hal::Backend>(src : *const u8,
                     byte_size: usize,
                     buffer_usage : buffer::Usage,
                     device: &B::Device,
                     mem_types:&Vec<MemoryType>,
                     comm_pool :&mut CommandPool<B,hal::Graphics>,
                     queue_group:&mut QueueGroup<B,hal::Graphics> ) -> Option<(B::Buffer,B::Memory)>
{
    let mut stag_buffer = device.create_buffer(byte_size as u64,
                                               buffer::Usage::TRANSFER_SRC | buffer_usage ).unwrap();

    let requirment = device.get_buffer_requirements(&stag_buffer) ;
    let mem_index = get_mem_type_index(requirment.type_mask,
                                       memory::Properties::COHERENT | memory::Properties::CPU_VISIBLE,
                                       mem_types).unwrap();

    let stag_mem = device.allocate_memory((mem_index as usize).into(),requirment.size).unwrap();
    device.bind_buffer_memory(&stag_mem,0,&mut stag_buffer).ok().unwrap();

    let ptr = device.map_memory(&stag_mem,0..(byte_size as u64) ).unwrap();
    copy(src,ptr, byte_size);
    device.unmap_memory(&stag_mem);

    let mut index_buffer = device.create_buffer(byte_size as u64,buffer::Usage::TRANSFER_DST | buffer_usage).unwrap();

    let requirment = device.get_buffer_requirements(&index_buffer);
    let mem_index = get_mem_type_index(requirment.type_mask,
                                       memory::Properties::DEVICE_LOCAL,mem_types).unwrap();
    let index_mem = device.allocate_memory((mem_index as usize).into(),requirment.size).unwrap();

    device.bind_buffer_memory(&index_mem,0,&mut index_buffer).ok().unwrap();

    let cp_cmd = {
        let mut cp_cmd:command::CommandBuffer<B,hal::Graphics,command::OneShot> = comm_pool.acquire_command_buffer::<command::OneShot>();
        cp_cmd.begin();

        let regions = command::BufferCopy{
            src:0,
            dst:0,
            size: byte_size as _
        };

        cp_cmd.copy_buffer(&stag_buffer,&index_buffer,&[regions]);

        cp_cmd.finish();
        cp_cmd
    };
    let fence = device.create_fence(false).unwrap();

    queue_group.queues[0].submit_nosemaphores(&[cp_cmd],Some(&fence));
    device.wait_for_fence(&fence,!0).ok().unwrap();

    device.destroy_fence(fence);

    device.free_memory(stag_mem);
    device.destroy_buffer(stag_buffer);

    Some((index_buffer,index_mem))
}

unsafe fn create_buffer<B:hal::Backend>(    byte_size: u64,
                                            buffer_usage : buffer::Usage,
                                            device: &B::Device,
                                            mem_types:&Vec<MemoryType> ) -> Option<(B::Buffer,B::Memory)>
{
    let mut buffer = device.create_buffer(byte_size,
                                               buffer_usage ).unwrap();

    let requirement = device.get_buffer_requirements(&buffer) ;
    let mem_index = get_mem_type_index(requirement.type_mask,
                                       memory::Properties::COHERENT | memory::Properties::CPU_VISIBLE,
                                       mem_types).unwrap();

    let mem = device.allocate_memory((mem_index as usize).into(),requirement.size).unwrap();
    device.bind_buffer_memory(&mem,0,&mut buffer).ok().unwrap();

    Some((buffer,mem))
}

fn update_uniform_buffer<B:hal::Backend>(device:&B::Device,mem:&B::Memory,tri:&Triangle,view_rot:Vector3<f32>,aspect:f32)
{
    let mut model = Matrix4::<f32>::from_scale(1.0f32);
    model = Matrix4::<f32>::from_angle_x(Rad(tri.rotate.x.to_radians())) * model;
    model = Matrix4::<f32>::from_angle_y(Rad(tri.rotate.y.to_radians())) * model;
    model = Matrix4::<f32>::from_angle_z(Rad(tri.rotate.z.to_radians())) * model;

    let mut view = Matrix4::<f32>::from_scale(1.0f32);
    view = Matrix4::<f32>::from_angle_x(Rad(view_rot.x.to_radians())) * view;
    view = Matrix4::<f32>::from_angle_y(Rad(view_rot.y.to_radians())) * view;
    view = Matrix4::<f32>::from_angle_z(Rad(view_rot.z.to_radians())) * view;
    view = Matrix4::<f32>::from_translation(tri.pos) * view;
    let ubo = Ubo{
        projection : cgmath::perspective(Deg(60.0),aspect,0.1f32,256.0f32),
        view,
        model
    };
    //let device = device as TOfB::Device;
    unsafe {
        let ptr = device.map_memory(mem,0..(size_of::<Ubo>() as u64)).unwrap();
        copy(&ubo as *const Ubo as *const _,ptr,size_of::<Ubo>());
        device.unmap_memory(&mem);
    }
}

fn create_descriptor_pool<B:hal::Backend>( device : &B::Device ) -> Result<B::DescriptorPool,hal::device::OutOfMemory>
{
    //let device = device as TOfB::Device;
    let descriptor_range_desc =
        hal::pso::DescriptorRangeDesc{
            ty : DescriptorType::UniformBuffer,
            count : 1
        };
    unsafe { device.create_descriptor_pool(1,&[descriptor_range_desc]) }
}

fn create_descriptor_set_layout<B: hal::Backend>( device : &B::Device ) -> Result<B::DescriptorSetLayout,hal::device::OutOfMemory>
{
    //let device = device as TOfB::Device;
    let binding = DescriptorSetLayoutBinding{
        binding : 0,
        ty : DescriptorType::UniformBuffer,
        count : 1,
        stage_flags : ShaderStageFlags::VERTEX,
        immutable_samplers : false
    };
    unsafe { device.create_descriptor_set_layout(&[binding],&[]) }
}

fn create_pipeline<B: hal::Backend>(device :&B::Device,pipeline_layout: &B::PipelineLayout,render_pass:&B::RenderPass)
    -> Result<B::GraphicsPipeline,hal::pso::CreationError>
{
    //let device = device as TOfB::Device;


    unsafe{
        let vs_module = device.create_shader_module(VERTEX_SHADER_DATA).unwrap();
        let fs_module = device.create_shader_module(FRAGMENT_SHADER_DATA).unwrap();

        let (vertex,fragment) = {
            let vs = EntryPoint{
                entry : "main",
                module : &vs_module,
                specialization : Default::default()
            };
            let fs = EntryPoint{
                entry : "main",
                module : &fs_module,
                specialization : Default::default()
            };
            (vs,fs)
        };

        let shaders = GraphicsShaderSet{
            vertex ,
            hull : None,
            domain : None,
            geometry : None,
            fragment : Some(fragment)
         };

        let subpass = Subpass{
            index : 0,
            main_pass : render_pass
        };

        let mut rasterizer = Rasterizer::FILL;
        rasterizer.front_face = FrontFace::Clockwise;

        let mut pipeline_desc = GraphicsPipelineDesc::new(shaders,
                                                      Primitive::TriangleList,
                                                      rasterizer,
                                                        pipeline_layout,
                                                        subpass);

        pipeline_desc.blender.targets.push(
            ColorBlendDesc::EMPTY
        );

        pipeline_desc.vertex_buffers.push(
            VertexBufferDesc{
                binding: 0,
                stride: size_of::<Vertex>() as _,
                rate : 0
            }
        );

        pipeline_desc.attributes.push(AttributeDesc{
            location: 0,
            binding : 0,
            element : Element::<Format>{
                format : Format::Rgb32Float,
                offset : 0
            }
        });

        pipeline_desc.attributes.push(AttributeDesc{
            location: 1,
            binding : 0,
            element : Element::<Format>{
                format : Format::Rgb32Float,
                offset : (size_of::<f32>() * 3) as _
            }
        });

        pipeline_desc.depth_stencil.depth = hal::pso::DepthTest::On { fun : hal::pso::Comparison::LessEqual,write:true };
        pipeline_desc.depth_stencil.stencil = hal::pso::StencilTest::Off;
        pipeline_desc.depth_stencil.depth_bounds = false;


        let res = device.create_graphics_pipeline(&pipeline_desc,None);
        device.destroy_shader_module(vs_module);
        device.destroy_shader_module(fs_module);
        res
    }
}


fn get_mem_type_index(mut type_mask:u64,properties:hal::memory::Properties,mem_types:&Vec<MemoryType>) -> Option<u64>
{
    let mem_type_count = mem_types.len() as _;
    let mut i = 0u64;
    loop{
        if i >= mem_type_count { break; }

        if (type_mask & 1) == 1
        {
            if (mem_types[i as usize].properties & properties) == properties
            {
                return Some(i);
            }
        }
        type_mask = type_mask >> 1;
        i += 1;
    }
    None
}
#[allow(dead_code)]
fn get_depth_format<B: hal::Backend>(physical_device:&B::PhysicalDevice) -> Option<Format>
{
    let expect_arr = [
        Format::D32FloatS8Uint,
        Format::D32Float,
        Format::D24UnormS8Uint,
        Format::D16UnormS8Uint,
        Format::D16Unorm
    ];
    //let physical_device = physical_device as TOfB::PhysicalDevice;

    for f in expect_arr.iter() {
        let properties = physical_device.format_properties(Some(*f));
        let res:u32 = unsafe{ std::mem::transmute(properties.optimal_tiling & ImageFeature::DEPTH_STENCIL_ATTACHMENT ) };
        if res != 0u32{
            return Some(*f);
        }
    }
    None
}

unsafe fn create_depth_stencil<B: hal::Backend>(device : &B::Device,depth_format:Format,w : u32,h :u32,mem_types:&Vec<MemoryType>) -> DepthStencil<B>
{
    //let device = device as TOfB::Device;
    let mut img = device.create_image(Kind::D2(w as _,h as _,1,1),
                        1,depth_format,
                        Tiling::Optimal,
                        Usage::DEPTH_STENCIL_ATTACHMENT | Usage::TRANSFER_SRC,
                        ViewCapabilities::empty()).unwrap();


    let requirment = device.get_image_requirements(&img);
    let mem_index = get_mem_type_index(requirment.type_mask,memory::Properties::DEVICE_LOCAL,mem_types).unwrap();
    let mem = device.allocate_memory((mem_index as usize).into(),requirment.size).unwrap();

    let range = SubresourceRange{
        aspects : Aspects::DEPTH | Aspects::STENCIL,
        layers : 0..1,
        levels : 0..1
    };
    device.bind_image_memory(&mem,0,&mut img).ok().unwrap();
    let view = device.create_image_view(&img,ViewKind::D2,depth_format,Swizzle::NO,range).unwrap();

    DepthStencil::<B>{
        image : img,
        memory : mem,
        view
    }
}

#[cfg(not(any(feature = "vulkan",feature = "dx12",feature = "gl")))]
fn main()
{
    println!("features must be one of vulkan dx12 gl!");
}