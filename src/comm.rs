
use hal::{  Adapter,
            Backend,
            Surface,
            QueueGroup,
            PhysicalDevice,
            MemoryType,
            Limits
};

use hal::adapter::{
    DeviceType
};

pub fn pick_adapter<B: Backend>(mut adapters: Vec<Adapter<B>>, surface: &B::Surface) -> Result<(B::Device, QueueGroup<B, hal::Graphics>,Adapter<B>), ()>
{
    let mut first_device: Option<(B::Device, QueueGroup<B, hal::Graphics>, Adapter<B>)> = None;
    let mut second_device: Option<(B::Device, QueueGroup<B, hal::Graphics>, Adapter<B>)> = None;


    for n in 0..adapters.len() {
        let adapter = adapters.remove(n);

        if let Ok(res) = adapter.open_with::<_, hal::Graphics>(1, |queue_family| {
            surface.supports_queue_family(queue_family) && adapter.info.device_type == DeviceType::DiscreteGpu
        }) {

            first_device = Some((res.0, res.1,adapter));
            break;
        }
        if let None = second_device {
            if let Ok(res) = adapter.open_with::<_, hal::Graphics>(1, |queue_family| {
                surface.supports_queue_family(queue_family)
            }) {
                second_device = Some((res.0, res.1,adapter));
            }
        }
    }

    if let Some(r) = first_device {
        return Ok(r);
    }

    if let Some(r) = second_device {
        return Ok(r);
    }
    return Err(());
}

