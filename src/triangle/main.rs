#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as bankend;
extern crate gfx_hal as hal;

extern crate winit;
extern crate image;
extern crate glsl_to_spirv;

extern crate learn_gfx;

use hal::{  Instance,
            Device,
            PhysicalDevice,
            pool,
            Surface
};
use hal::pass::{
    Attachment,
    AttachmentOps,
    AttachmentLoadOp,
    AttachmentStoreOp,
    SubpassDesc,
    AttachmentLayout
};

use hal::format::{
    Format
};

use hal::image::Layout;

use hal::format::ChannelType;

use learn_gfx::comm::pick_adapter;


const W:u32 = 800;
const H:u32 = 600;
const TITLE:&'static str = "triangle";
//參考 https://github.com/Mistodon/gfx-hal-tutorials/blob/master/src/bin/part00-triangle.rs
#[cfg(feature = "vulkan")]
fn main()
{
    let mut events_loop = winit::EventsLoop::new();
    let wb = winit::WindowBuilder::new()
        .with_dimensions(winit::dpi::LogicalSize::new( W as _,H as _ ))
        .with_title(TITLE);

    let (_window,_instance,surface,adapters) = {
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

    let command_pool = unsafe { device.create_command_pool_typed(&queue_group,pool::CommandPoolCreateFlags::empty()) }.unwrap();

    let physical_device = &(adapter.physical_device);

    let (caps,formats,..) = surface.compatibility(physical_device);

    println!("caps = {:?}",caps);
    println!("formats = {:?}",formats);

    let format = if let Some(fs) = formats{
        fs.into_iter().find(|it|{
            it.base_format().1 == ChannelType::Srgb
        }).unwrap()
    }else{
        Format::Rgba8Srgb
    };

    {
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
    }

    device.wait_idle().unwrap();
    unsafe {
        device.destroy_command_pool(command_pool.into_raw());
    }
}



#[cfg(not(feature = "vulkan"))]
fn main()
{
    println!("feature must be vulkan!");
}