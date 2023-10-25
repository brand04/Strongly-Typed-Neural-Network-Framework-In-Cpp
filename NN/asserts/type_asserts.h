#pragma once
#include "./assert_t.h"
#include "./bulk_assert_t.h"
#include "../shapes/shape_concept.h"
#include "../shapes/compiler_operations/pad_shape.h"
#include "../dtypes/concept.h"
#include "../devices/device.h"

#include <concepts>
namespace NN {
	namespace Asserts {

		/// <summary>
		/// Runs a static assertion on Shape (when passed to struct_assert<...>) that ensures the supplied argument is a shape
		/// </summary>
		/// <typeparam name="Shape">Shape to assert on</typeparam>
		template<typename Shape>
		struct AssertShape : Assert<Shapes::IsShape<Shape>> {
			static_assert(Shapes::IsShape<Shape>, "Fundamental Error - Expected a Shape");
		};

		/// <summary>
		/// Runs a static assertion on the arguments, ensuring that the types ShapeL and ShapeR are equal
		/// </summary>
		/// <typeparam name="ShapeL">A shape equal to ShapeR</typeparam>
		/// <typeparam name="ShapeR">A shape equal to ShapeL</typeparam>
		template<typename ShapeL, typename ShapeR>
		struct AssertSameShape : Assert <struct_assert<BulkAssert<AssertShape, ShapeL,ShapeR>> && std::same_as<ShapeL, ShapeR>> {
			static_assert(std::same_as<ShapeL, ShapeR>, "Expected the same shape");
		};

		/// <summary>
		/// Runs a static assertion on the arguments, ensuring that the types ShapeL and ShapeR are equivalent (two shapes which, when padded with a 1 to the highest of their dimensions, are equal)
		/// </summary>
		/// <typeparam name="ShapeL">A shape equivalent to ShapeR</typeparam>
		/// <typeparam name="ShapeR">A shape equivalent to ShapeL</typeparam>
		template<typename ShapeL, typename ShapeR>
		struct AssertEquivalentShapes : Assert<struct_assert<BulkAssert<AssertShape,ShapeL,ShapeR>> && struct_assert<AssertSameShape<Shapes::PadToMax<ShapeL, ShapeR>, Shapes::PadToMax<ShapeR, ShapeL>>>> {
			static_assert(std::is_same_v < Shapes::PadToMax<ShapeL, ShapeR>, Shapes::PadToMax<ShapeR, ShapeL>>, "Expected equivalent shapes: two shapes which, when padded with 1 to the highest of their dimensions, are equal");
		};

		/// <summary>
		/// Runs a static assertion on Device (when passed to struct_assert<...>) that ensures the supplied argument is a Device
		/// </summary>
		/// <typeparam name="Device">A Device</typeparam>
		template<typename Device>
		struct AssertDevice : Assert<Devices::Device<Device>> {
			static_assert(Devices::Device<Device>, "Fundamental Error - Expected a Device");
		};

		/// <summary>
		/// Runs a static assertion on DeviceL and DeviceR (when passed to struct_assert<...>) that ensures DeviceL and DeviceR are different devices
		/// </summary>
		/// <typeparam name="DeviceL">A Device not equal to DeviceR</typeparam>
		/// <typeparam name="DeviceR">A Device not equal to DeviceL</typeparam>
		template<typename DeviceL, typename DeviceR>
		struct AssertDifferentDevices : BulkAssert<AssertDevice, DeviceL, DeviceR > {
			static_assert(!std::same_as<DeviceL, DeviceR>, "Expected different devices");
			static consteval bool stassert() {
				return !std::same_as<DeviceL, DeviceR>&& struct_assert<BulkAssert<AssertDevice, DeviceL, DeviceR>>;
			}
		};

		/// <summary>
		/// Runs a static assertion on DeviceL and DeviceR (when passed to struct_assert<...>) that ensures DeviceL and DeviceR are the same device
		/// </summary>
		/// <typeparam name="DeviceL">A Device equal to DeviceR</typeparam>
		/// <typeparam name="DeviceR">A Device equal to DeviceL</typeparam>
		template<typename DeviceL, typename DeviceR>
		struct AssertSameDevices : Assert<> {
			static_assert(std::same_as<DeviceL, DeviceR>, "Expected the same device");
			static consteval bool stassert() {
				return std::same_as<DeviceL, DeviceR> && struct_assert<BulkAssert<AssertDevice, DeviceL, DeviceR>>;
			}
		};

		/// <summary>
		/// Runs a static assertion on the types that ensures TypeL and typeR are not the same type
		/// </summary>
		/// <typeparam name="TypeL"></typeparam>
		/// <typeparam name="TypeR"></typeparam>
		template<typename TypeL, typename TypeR>
		struct AssertDifferentTypes : Assert<!std::same_as<TypeL, TypeR>> {
			static_assert(!std::same_as<TypeL, TypeR>, "Expected different types");
		};


		//Asserts the argument is a Dtype
		template<typename Dtype>
		struct AssertDtype : Assert<Dtypes::Dtype<Dtype>> {
			static_assert(Dtypes::Dtype<Dtype>, "Fundamental Error - Expected a Dtype");
		};

		/// <summary>
		/// Asserts that the given layer parameters make sense
		/// </summary>
		/// <typeparam name="InputDtype">The Input Datatype of the layer</typeparam>
		/// <typeparam name="OutputShape">The Output Shape of the layer</typeparam>
		/// <typeparam name="InputDevice">The Input Device of the layer</typeparam>
		/// <typeparam name="OutputDtype">The Output Datatype of the layer</typeparam>
		/// <typeparam name="OutputDevice">The Output Device of the layer</typeparam>
		/// <typeparam name="InputShape">The Input Shape of the layer</typeparam>
		template<typename InputDtype, typename InputShape, typename InputDevice, typename OutputDtype, typename OutputShape, typename OutputDevice>
		struct AssertLayerParameters : Assert<> {
			static consteval bool stassert() {
				return struct_assert<BulkAssert<AssertDevice, InputDevice, OutputDevice>>
					& struct_assert<BulkAssert<AssertShape, InputShape, OutputShape>>
					& struct_assert<BulkAssert<AssertDtype, InputDtype, OutputDtype>>;
			}
		};
	}
}