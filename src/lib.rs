use std::ptr::NonNull;

use gpu_alloc_types::{
    AllocationFlags, DeviceMapError, DeviceProperties, MappedMemoryRange, MemoryDevice, MemoryHeap,
    MemoryPropertyFlags, MemoryType, OutOfMemory,
};
use smallvec::SmallVec;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::InstanceV1_1;

/// Vulkan device extension trait which wraps its reference into memory device.
pub trait AsMemoryDevice {
    /// Returns a [`MemoryDevice`] wrapper.
    fn as_memory_device(&self) -> &VulkanaliaMemoryDevice;
}

impl AsMemoryDevice for Device {
    fn as_memory_device(&self) -> &VulkanaliaMemoryDevice {
        VulkanaliaMemoryDevice::wrap(self)
    }
}

/// A wrapper around Vulkan device which implements [`MemoryDevice`].
#[repr(transparent)]
pub struct VulkanaliaMemoryDevice {
    device: Device,
}

impl VulkanaliaMemoryDevice {
    pub fn wrap(device: &Device) -> &Self {
        unsafe {
            // SAFETY: `VulkanaliaMemoryDevice` has the same layout as `Device`
            &*(device as *const Device).cast::<Self>()
        }
    }
}

impl MemoryDevice<vk::DeviceMemory> for VulkanaliaMemoryDevice {
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn allocate_memory(
        &self,
        size: u64,
        memory_type: u32,
        flags: AllocationFlags,
    ) -> Result<vk::DeviceMemory, OutOfMemory> {
        assert!((flags & !(AllocationFlags::DEVICE_ADDRESS)).is_empty());

        let mut info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type);

        let mut info_flags;

        if flags.contains(AllocationFlags::DEVICE_ADDRESS) {
            info_flags = vk::MemoryAllocateFlagsInfo::builder()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            info = info.push_next(&mut info_flags);
        }

        match self.device.allocate_memory(&info, None) {
            Ok(memory) => Ok(memory),
            Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => Err(OutOfMemory::OutOfDeviceMemory),
            Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => Err(OutOfMemory::OutOfHostMemory),
            Err(e) => panic!("Unexpected Vulkan error: {e}"),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn deallocate_memory(&self, memory: vk::DeviceMemory) {
        self.device.free_memory(memory, None);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn map_memory(
        &self,
        memory: &mut vk::DeviceMemory,
        offset: u64,
        size: u64,
    ) -> Result<std::ptr::NonNull<u8>, DeviceMapError> {
        match self
            .device
            .map_memory(*memory, offset, size, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => {
                Ok(NonNull::new(ptr as *mut u8)
                    .expect("Pointer to memory mapping must not be null"))
            }
            Err(vk::ErrorCode::OUT_OF_DEVICE_MEMORY) => Err(DeviceMapError::OutOfDeviceMemory),
            Err(vk::ErrorCode::OUT_OF_HOST_MEMORY) => Err(DeviceMapError::OutOfHostMemory),
            Err(vk::ErrorCode::MEMORY_MAP_FAILED) => Err(DeviceMapError::MapFailed),
            Err(e) => panic!("Unexpected Vulkan error: {e}"),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn unmap_memory(&self, memory: &mut vk::DeviceMemory) {
        self.device.unmap_memory(*memory);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn invalidate_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .invalidate_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk::MappedMemoryRange::builder()
                            .memory(*range.memory)
                            .offset(range.offset)
                            .size(range.size)
                    })
                    .collect::<SmallVec<[_; 4]>>(),
            )
            .map_err(|e| match e {
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk::ErrorCode::OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                e => panic!("Unexpected Vulkan error: {e}"),
            })
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self)))]
    unsafe fn flush_memory_ranges(
        &self,
        ranges: &[MappedMemoryRange<'_, vk::DeviceMemory>],
    ) -> Result<(), OutOfMemory> {
        self.device
            .flush_mapped_memory_ranges(
                &ranges
                    .iter()
                    .map(|range| {
                        vk::MappedMemoryRange::builder()
                            .memory(*range.memory)
                            .offset(range.offset)
                            .size(range.size)
                    })
                    .collect::<SmallVec<[_; 4]>>(),
            )
            .map_err(|e| match e {
                vk::ErrorCode::OUT_OF_DEVICE_MEMORY => OutOfMemory::OutOfDeviceMemory,
                vk::ErrorCode::OUT_OF_HOST_MEMORY => OutOfMemory::OutOfHostMemory,
                e => panic!("Unexpected Vulkan error: {e}"),
            })
    }
}

/// Collects device properties from vulkanalia's `Instance` for the specified
/// physical device, required to create `GpuAllocator`.
///
///
/// # Safety
///
/// The following must be true:
/// - `version` must not be higher than the `api_version` of the `instance`.
/// - `physical_device` must be queried from an [`Instance`] associated with this `instance`.
/// - Even if returned properties' field `buffer_device_address` is set to true,
///   feature `PhysicalDeviceBufferDeviceAddressFeatures::buffer_derive_address`
///   must be enabled explicitly on device creation and extension "VK_KHR_buffer_device_address"
///   for Vulkan prior 1.2.
///   Otherwise the field must be set to false before passing to `GpuAllocator::new`.
pub unsafe fn device_properties(
    instance: Instance,
    version: u32,
    physical_device: vk::PhysicalDevice,
) -> VkResult<DeviceProperties<'static>> {
    struct ExtInfo {
        buffer_device_address: bool,
        max_memory_allocation_size: u64,
    }

    let memory_properties = instance.get_physical_device_memory_properties(physical_device);

    // Determine what to fetch by instance version and device features
    let (query_props, query_features) = {
        let mut required_extensions: [_; 3] = [
            Some(&vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name),
            Some(&vk::KHR_MAINTENANCE3_EXTENSION.name),
            Some(&vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION.name),
        ];

        match vk::version_minor(version) {
            0 => {}
            1 => {
                // `get_physical_device_features2` is mandatory since 1.1
                required_extensions[0] = None;
                // `max_memory_allocation_size` is mandatory since 1.1
                required_extensions[1] = None;
            }
            _ => {
                // `PhysicalDeviceBufferDeviceAddressFeatures` is mandatory since 1.2
                required_extensions = [None, None, None];
            }
        }

        if required_extensions.iter().any(Option::is_some) {
            let extensions =
                instance.enumerate_device_extension_properties(physical_device, None)?;

            // Check whether all required extensions are supported
            let mut to_find = required_extensions.len();
            'extensions: for extension in extensions {
                if to_find == 0 {
                    break 'extensions;
                }

                for required in required_extensions.iter_mut() {
                    if let Some(name) = *required {
                        if name == &extension.extension_name {
                            *required = None;
                            to_find -= 1;
                            continue 'extensions;
                        }
                    }
                }
            }

            let [props2_ext, limits_ext, bda_ext] = required_extensions;
            (
                props2_ext.is_none() && limits_ext.is_none(),
                props2_ext.is_none() && bda_ext.is_none(),
            )
        } else {
            (true, true)
        }
    };

    let mut ext_info = ExtInfo {
        buffer_device_address: false,
        max_memory_allocation_size: u64::MAX,
    };

    // Query physical device properties
    let limits = if query_props {
        let mut properties = vk::PhysicalDeviceProperties2::builder();
        let mut maintenance3 = vk::PhysicalDeviceMaintenance3Properties::builder();
        properties = properties.push_next(&mut maintenance3);
        instance.get_physical_device_properties2(physical_device, &mut properties);

        let limits = properties.properties.limits;
        ext_info.max_memory_allocation_size = maintenance3.max_memory_allocation_size;
        limits
    } else {
        instance
            .get_physical_device_properties(physical_device)
            .limits
    };

    // Query physical device features
    if query_features {
        let mut features = vk::PhysicalDeviceFeatures2::builder();
        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::builder();
        features = features.push_next(&mut bda_features);
        instance.get_physical_device_features2(physical_device, &mut features);

        ext_info.buffer_device_address = bda_features.buffer_device_address != 0;
    };

    // Make device properties
    Ok(DeviceProperties {
        memory_types: memory_properties.memory_types
            [..memory_properties.memory_type_count as usize]
            .iter()
            .map(|memory_type| MemoryType {
                props: memory_properties_from(memory_type.property_flags),
                heap: memory_type.heap_index,
            })
            .collect(),
        memory_heaps: memory_properties.memory_heaps
            [..memory_properties.memory_heap_count as usize]
            .iter()
            .map(|memory_heap| MemoryHeap {
                size: memory_heap.size,
            })
            .collect(),
        max_memory_allocation_count: limits.max_memory_allocation_count,
        max_memory_allocation_size: ext_info.max_memory_allocation_size,
        non_coherent_atom_size: limits.non_coherent_atom_size,
        buffer_device_address: ext_info.buffer_device_address,
    })
}

/// Maps `vulkanalia`'s `MemoryPropertyFlags` to `gpu-alloc-types`.
pub fn memory_properties_from(props: vk::MemoryPropertyFlags) -> MemoryPropertyFlags {
    let mut result = MemoryPropertyFlags::empty();
    if props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
        result |= MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
        result |= MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
        result |= MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}

/// Maps `gpu-alloc-types`' `MemoryPropertyFlags` to `vulkanalia`.
pub fn memory_properties_to(props: MemoryPropertyFlags) -> vk::MemoryPropertyFlags {
    let mut result = vk::MemoryPropertyFlags::empty();
    if props.contains(MemoryPropertyFlags::DEVICE_LOCAL) {
        result |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
    }
    if props.contains(MemoryPropertyFlags::HOST_VISIBLE) {
        result |= vk::MemoryPropertyFlags::HOST_VISIBLE;
    }
    if props.contains(MemoryPropertyFlags::HOST_COHERENT) {
        result |= vk::MemoryPropertyFlags::HOST_COHERENT;
    }
    if props.contains(MemoryPropertyFlags::HOST_CACHED) {
        result |= vk::MemoryPropertyFlags::HOST_CACHED;
    }
    if props.contains(MemoryPropertyFlags::LAZILY_ALLOCATED) {
        result |= vk::MemoryPropertyFlags::LAZILY_ALLOCATED;
    }
    result
}
