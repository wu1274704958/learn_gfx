#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as bankend;

//#[cfg(feature = "dx12")]
//extern crate gfx_backend_dx12 as bankend;

extern crate gfx_hal as hal;
extern crate winit;
extern crate learn_gfx;

type TOfB = bankend::Backend;

use winit::WindowEvent;
use hal::{ Instance,PhysicalDevice,Device,Surface,SurfaceCapabilities,SwapchainConfig,memory};
use hal::window::{ Extent2D,Backbuffer };
use hal::image::{ViewKind,Extent,SubresourceRange,Layout,Access};
use hal::buffer;
use hal::pass::{Attachment,
                AttachmentOps,
                AttachmentLoadOp,
                AttachmentStoreOp,
                SubpassDesc,
                AttachmentLayout,
                SubpassDependency,
                SubpassRef,
                Subpass
};
use hal::pso::*;
use hal::format::{Format,ChannelType,Swizzle,Aspects};
use hal::pool::{CommandPoolCreateFlags};
use hal::adapter::{ MemoryType };
use learn_gfx::comm::pick_adapter;
use std::mem::size_of;
use std::ptr::copy;

const W:u32 = 800;
const H:u32 = 600;
const TITLE:&'static str = "triangle";

//shader data
const VERTEX_SHADER_DATA :&[u8] = include_bytes!("../../data/triangle2/triangle.vert.spv");
const FRAGMENT_SHADER_DATA :&[u8] = include_bytes!("../../data/triangle2/triangle.frag.spv");

#[cfg(any(feature = "vulkan",feature = "dx12"))]
fn main()
{
    let events_loop = winit::EventsLoop::new();
    let wb = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new(W as f64,H as f64))
        .with_title(TITLE);

    let (window,instance,mut surface,mut adapters) = {
        let window = wb.build(&events_loop).unwrap();
        let instance = bankend::Instance::create(TITLE,1);
        let surface = instance.create_surface(&window);
        let adapters = instance.enumerate_adapters();

        (window,instance,surface,adapters)
    };

    let (device,mut queue_group,adapter) = if let Ok(res) = pick_adapter(adapters,&surface)
    {
        res
    }else {
        panic!("failed to pick a adapter!")
    };

    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let limits = adapter.physical_device.limits();
    let mut command_pool = unsafe {
        device.create_command_pool_typed(&queue_group,CommandPoolCreateFlags::empty())
    }.unwrap();

    let physical_device = &adapter.physical_device;

    let (caps,formats,..) = surface.compatibility(physical_device);

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

    let render_pass = create_render_pass::<bankend::Backend>(&device,format).ok().unwrap();
    let (mut swap_chain,extent,image_views,frame_buffers) = create_swapchain::<bankend::Backend>(&device,
                                                                             &mut surface,
                                                                             &render_pass,
                                                                             &caps,format,
                                                                             W ,H,None);

    unsafe {
        device.destroy_swapchain(swap_chain);
        device.destroy_command_pool(command_pool.into_raw());
        device.destroy_render_pass(render_pass);
        for iv in image_views{
            device.destroy_image_view(iv);
        }
        for fb in frame_buffers{
            device.destroy_framebuffer(fb);
        }
    }
}

fn create_render_pass<B : hal::Backend>(device:&B::Device,format:Format) -> Result<B::RenderPass, hal::device::OutOfMemory>
{
    let color_attachment = Attachment{
        format : Some(format),
        samples: 1,
        ops: AttachmentOps::new(AttachmentLoadOp::Clear,AttachmentStoreOp::Store),
        stencil_ops : AttachmentOps::DONT_CARE,
        layouts:  Layout::Undefined..Layout::Present
    };
    let sub_pass_desc = SubpassDesc{
        colors : &[(0,Layout::ColorAttachmentOptimal)],
        depth_stencil : None,
        inputs : &[],
        resolves : &[],
        preserves : &[]
    };
    let sub_pass_dependency = SubpassDependency{
        passes : SubpassRef::External .. SubpassRef::Pass(0),
        stages : PipelineStage::BOTTOM_OF_PIPE ..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
        accesses: Access::MEMORY_READ..(Access::COLOR_ATTACHMENT_WRITE | Access::COLOR_ATTACHMENT_READ)
    };

    unsafe { device.create_render_pass(&[color_attachment],
                                       &[sub_pass_desc],
                                       &[sub_pass_dependency]) }
}

fn create_swapchain<B:hal::Backend>(device:&B::Device,
                                    surface:& mut B::Surface,
                                    render_pass:&B::RenderPass,
                                    caps:&SurfaceCapabilities,
                                    format:Format,w:u32,h:u32,
                                    old_swapchain:Option<B::Swapchain>) -> (B::Swapchain,Extent,Vec<B::ImageView>,Vec<B::Framebuffer>)
{
    let swapchain_config = SwapchainConfig::from_caps(caps,format,
                                                      Extent2D{ width:w,height:h });
    let extent = swapchain_config.extent.to_extent();

    let (mut swapchain, backbuffer) = unsafe { device.create_swapchain(surface,swapchain_config,old_swapchain).unwrap()};

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
                unsafe {device.create_framebuffer(render_pass,vec![it],extent).unwrap()}
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

unsafe fn create_vertex_buffer<B:hal::Backend>(device: &B::Device,mem_types:&Vec<MemoryType>,comm_pool :&B::CommandPool) -> Option<(B::Buffer,B::Memory)>
{
    let vertices = vec![
        Vertex{ pos:[  1.0f32,  1.0f32, 0.0f32 ],   color:[ 1.0f32, 0.0f32, 0.0f32  ] },
        Vertex{ pos:[  -1.0f32,  1.0f32, 0.0f32 ] , color:[ 0.0f32, 1.0f32, 0.0f32  ] },
        Vertex{ pos:[  0.0f32, -1.0f32, 0.0f32 ] , color:[ 0.0f32, 0.0f32, 1.0f32  ] }
    ];

    let vertex_byte_size = size_of::<Vertex>() as u64 * 3;
    //let device = (device as TOfB::Device);
    let mut stag_buffer = device.create_buffer(vertex_byte_size,
                                                             buffer::Usage::TRANSFER_SRC | buffer::Usage::VERTEX ).unwrap();

    let requirment = device.get_buffer_requirements(&stag_buffer) ;
    let mem_index = get_mem_type_index(requirment.type_mask,
                                       memory::Properties::COHERENT | memory::Properties::CPU_VISIBLE,
    mem_types).unwrap();

    let stag_mem = device.allocate_memory((mem_index as usize).into(),requirment.size).unwrap();
    device.bind_buffer_memory(&stag_mem,0,&mut stag_buffer);

    let ptr = device.map_memory(&stag_mem,0..vertex_byte_size).unwrap();
    copy(vertices.as_ptr() as *const u8,ptr, size_of::<Vertex>()  * 3);
    device.unmap_memory(&stag_mem);

    let mut vertexBuffer = device.create_buffer(vertex_byte_size,buffer::Usage::TRANSFER_DST | buffer::Usage::VERTEX).unwrap();

    let requirment = device.get_buffer_requirements(&vertexBuffer);
    let mem_index = get_mem_type_index(requirment.type_mask,
                                                    memory::Properties::DEVICE_LOCAL,mem_types).unwrap();
    let vertexMem = device.allocate_memory((mem_index as usize).into(),requirment.size).unwrap();

    device.bind_buffer_memory(&vertexMem,0,&mut vertexBuffer);

    None

}

fn get_mem_type_index(type_mask:u64,properties:hal::memory::Properties,mem_types:&Vec<MemoryType>) -> Option<u64>
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

        i += 1;
    }
    None
}

#[cfg(not(any(feature = "vulkan",feature = "dx12")))]
fn main()
{
    println!("features must be vulkan or dx12!");
}