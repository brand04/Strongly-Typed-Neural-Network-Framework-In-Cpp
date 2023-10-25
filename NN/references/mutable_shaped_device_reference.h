#pragma once

#include "../shapes/shape_concept.h"
#include "../devices/device.h"
#include "./mutable_reference.h"
#include "./immutable_shaped_device_reference.h"
namespace NN {
	namespace References {
		/// <summary>
		/// A wrapper around an MutableReference - which itself is a wrapper around a T* 
		/// </summary>
		/// <typeparam name="T">The underlying type</typeparam>
		/// <typeparam name="Shape">The Shape of the data stored</typeparam>
		/// <typeparam name="Device">The Device on which the data is stored</typeparam>
		template<typename T, Shapes::IsShape Shape, Devices::Device Device>
		struct MutableShapedDeviceReference : public MutableReference<T> {
			using MutableReference<T>::MutableReference;

			inline operator ImmutableShapedDeviceReference<T, Shape, Device>() const {
				return ImmutableShapedDeviceReference<T, Shape, Device>(this->ptr);
			}

			inline ImmutableShapedDeviceReference<T, Shape, Device> asImmutable() const {
				return ImmutableShapedDeviceReference<T, Shape, Device>(this->ptr);
			}
		};
	}
}