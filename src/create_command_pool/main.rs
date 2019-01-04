#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as bankend;
extern crate gfx_hal as hal;

extern crate winit;
extern crate image;
extern crate glsl_to_spirv;

extern crate learn_gfx;

use hal::{  Instance,
            Device,
            pool
};

use learn_gfx::comm::pick_adapter;


const W:u32 = 800;
const H:u32 = 600;
const TITLE:&'static str = "create command pool";

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

    let res = pick_adapter(adapters,surface);
    let (device,mut queue_group,memory_types,limits) = if let Ok(adapter) = res{
        adapter
    }else {
        panic!("can not pick a adapter!");
    };

    let command_pool = unsafe { device.create_command_pool_typed(&queue_group,pool::CommandPoolCreateFlags::empty()) }.unwrap();

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