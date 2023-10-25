#pragma once

#include "../shapes/shape_concept.h"
#include "../devices/device.h"
#include "immutable_reference.h"


namespace NN {
	namespace References {
		/// <summary>
		/// A wrapper around an ImmutableReference - which itself is a wrapper around a T const *const 

		/// </summary>
		/// <typeparam name="T">The underlying type</typeparam>
		/// <typeparam name="Shape">The Shape of the data stored</typeparam>
		/// <typeparam name="Device">The Device on which the data is stored</typeparam>
		template<typename T, Shapes::IsShape Shape, Devices::Device Device>
		struct ImmutableShapedDeviceReference : public ImmutableReference<T> {
			using ImmutableReference<T>::ImmutableReference;

			
		};


	}
}