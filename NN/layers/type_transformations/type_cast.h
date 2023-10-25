#pragma once
#include <fstream>
#include "../abstract/layer_box.h"
#include "../../devices/includes.h"
#include "../../dtypes/concept.h"
#include "../../shapes/shape_concept.h"

#include "../../kernels/cast.cuh"
#include "../../asserts/type_asserts.h"

#include "../../helpers/string_literal.h"
#include <concepts>
namespace NN {
	namespace Layers {

		/// <summary>
		/// A layer that casts from one type to another
		/// The input and output each have separate storage
		/// </summary>
		/// <typeparam name="InputDtype">Input Dtype</typeparam>
		/// <typeparam name="OutputDtype">Output Dtype</typeparam>
		/// <typeparam name="Device">Input and Output Device</typeparam>
		/// <typeparam name="threads">number of threads</typeparam>
		/// <typeparam name="Shape">Input and Output Shape</typeparam>
		template<typename InputDtype, typename OutputDtype, Shapes::IsShape Shape, Devices::Device Device, unsigned int threads>
		class Cast;

		//A layer, with the sole operation of casting data from one type to another - reserving the storage permanently
		template<typename InputDtype, typename OutputDtype, Shapes::IsShape Shape, Devices::Device Device, unsigned int threads>
		class Cast : public Abstract::LayerBox<InputDtype, Shape, Device, OutputDtype, Shape, Device, threads> {
			
			using InputType = typename InputDtype::Type;
			using OutputType = typename OutputDtype::Type;
			static_assert(struct_assert<Asserts::AssertDifferentTypes<InputType, OutputType>>, "Failed to create Cast layer"); //assert that the devices are not equal
		public:

			static constexpr StringLiteral name = StringLiteral("Cast");

			void threadForward(
				References::ImmutableShapedDeviceReference<InputType, Shape, Device> inputs,
				References::MutableShapedDeviceReference<OutputType, Shape, Device> outputs,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				Kernels::Cast::cast(inputs, outputs, threadContext);
				threadContext.synchronize();

			}

			void threadBackward(
				References::ImmutableShapedDeviceReference<InputType, Shape, Device> inputs,
				References::ImmutableShapedDeviceReference<OutputType, Shape, Device> outputs,
				References::MutableShapedDeviceReference<OutputType, Shape, Device> postNodeDeltas,
				References::MutableShapedDeviceReference<InputType, Shape, Device> preNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				Kernels::Cast::cast<Shape,OutputType, InputType, Device>(postNodeDeltas, preNodeDeltas, threadContext);
				threadContext.synchronize();
			}


			void threadModifyWeights(
				References::ImmutableShapedDeviceReference<InputType, Shape, Device> inputs,
				References::ImmutableShapedDeviceReference<OutputType, Shape, Device> postNodeDeltas,
				const Threads::ThreadContext& threadContext
			) {} //no action required - this layer does not store any weights

			static void saveAsBytes(Cast<InputDtype, OutputDtype, Shape, Device, threads>& layer, std::ofstream& file, Threads::ThreadContext& ctx) {
				//no weights - no action required
			}
			static void readFromBytes(Cast<InputDtype, OutputDtype, Shape, Device, threads>& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
				//no weights - no action required
			}
			using Abstract::LayerBox<InputDtype, Shape, Device, OutputDtype, Shape, Device, threads>::LayerBox;


			void setWeights(uint8_t*& handle, const Threads::ThreadContext& threadContext) {} //no action required
			void initializeWeights(const Threads::ThreadContext& threadContext) {} //no action required
			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, const Threads::ThreadContext& threadContext) const {} //no action required

		};



	}
}