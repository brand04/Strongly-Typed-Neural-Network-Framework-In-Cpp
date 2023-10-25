#pragma once
#include <fstream>
#include "../abstract/layer_box.h"
#include "../../devices/includes.h"
#include "../../dtypes/concept.h"
#include "../../shapes/shape_concept.h"

#include "../../asserts/type_asserts.h"

#include "../../helpers/string_literal.h"
#include <concepts>
namespace NN {
	namespace Layers {
		
		template<Dtypes::Dtype Dtype, Shapes::IsShape Shape, Devices::Device InputDevice, Devices::Device OutputDevice, unsigned int threads>
		class DeviceChange;

		//A layer, with the sole operation of transferring data from one device to another - reserving the storage permanently
		template<Dtypes::Dtype Dtype, Shapes::IsShape Shape, Devices::Device InputDevice, Devices::Device OutputDevice, unsigned int threads>
		class DeviceChange : public Abstract::LayerBox<Dtype,Shape,InputDevice,Dtype,Shape,OutputDevice,threads> {
			static_assert(struct_assert<Asserts::AssertDifferentDevices<InputDevice, OutputDevice>>, "Failed to create DeviceChange layer"); //assert that the devices are not equal
			using dtype = typename Dtype::Type;
		public:

			static constexpr StringLiteral name = StringLiteral("DeviceChange");

			void threadForward(
				References::ImmutableShapedDeviceReference<dtype, Shape, InputDevice> inputs,
				References::MutableShapedDeviceReference<dtype, Shape, OutputDevice> outputs,
				Threads::ThreadContext& threadContext
			) {
				if (Devices::CUDADevice<OutputDevice>) {
					threadContext.prepareCuda(); // initializes cuda if required
				}
				threadContext.synchronize();
				//simply a direct copy of data from the input store to the output store, allowing the store to handle the specifics of the transfer
				this->inputTensor.link->data.copyThreadTo(this->outputTensor.link->data, threadContext);
				threadContext.synchronize();
			}

			void threadBackward(
				References::ImmutableShapedDeviceReference<dtype, Shape, InputDevice> inputs,
				References::ImmutableShapedDeviceReference<dtype, Shape, OutputDevice> outputs,
				References::MutableShapedDeviceReference<dtype, Shape, OutputDevice> postNodeDeltas,
				References::MutableShapedDeviceReference<dtype, Shape, InputDevice> preNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				// this->getOutputTensor().link->data.copyThread(this->getInputTensor().link->data, threadContext); - would be required except that the two are guarenteed to be the same
				this->outputTensor.link->deltas.copyThreadTo(this->inputTensor.link->deltas, threadContext); //copy deltas back since these will have changed
				threadContext.synchronize();
			}


			void threadModifyWeights(
				References::ImmutableShapedDeviceReference<dtype, Shape, InputDevice> inputs,
				References::ImmutableShapedDeviceReference<dtype, Shape, OutputDevice> postNodeDeltas,
				const Threads::ThreadContext& threadContext
			) {} //no action required - this layer does not store any weights

			static void saveAsBytes(DeviceChange<Dtype,Shape, InputDevice, OutputDevice, threads>& layer, std::ofstream& file, Threads::ThreadContext& ctx) {
				//no weights - no action required
			}
			static void readFromBytes(DeviceChange<Dtype, Shape, InputDevice, OutputDevice, threads>& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
				if (Devices::CUDADevice<OutputDevice>) {
					 ctx.prepareCuda();
				}
			}
			using Abstract::LayerBox<Dtype, Shape, InputDevice, Dtype, Shape, OutputDevice, threads>::LayerBox;


			void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext) {
				if (Devices::CUDADevice<OutputDevice>) {
					threadContext.prepareCuda();
				}
			}
			void initializeWeights(Threads::ThreadContext& threadContext) {
				if (Devices::CUDADevice<OutputDevice>) {
					threadContext.prepareCuda();
				}
			}
			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, const Threads::ThreadContext& threadContext) const {} //no action required

		};


		
	}
}