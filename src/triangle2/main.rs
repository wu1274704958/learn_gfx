#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as bankend;

//#[cfg(feature = "dx12")]
//extern crate gfx_backend_dx12 as bankend;

extern crate gfx_hal as hal;
extern crate winit;
extern crate learn_gfx;

use winit::WindowEvent;
use hal::{ Instance,PhysicalDevice,Device,Surface,SurfaceCapabilities,SwapchainConfig};
use hal::window::{ Extent2D,Backbuffer };
use hal::image::{ViewKind,Extent,SubresourceRange,Layout,Access};

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
use learn_gfx::comm::pick_adapter;


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

#[cfg(not(any(feature = "vulkan",feature = "dx12")))]
fn main()
{
    println!("features must be vulkan or dx12!");
}