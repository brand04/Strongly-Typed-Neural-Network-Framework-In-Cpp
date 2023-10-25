#pragma once

#include "../../tensors/tensors.h"
#include "../interfaces/i_layer.h"
#include "../concepts/layer_concept.h"
#include <stdlib.h>
#include <thread>


namespace NN {
	namespace Layers {
		namespace Abstract {
			
			//Abstract layer implementation - stores the location of its inputs and outputs
			//template<Dtypes::Dtype InputDtype, Shapes::IsShape InputShape, Devices::Device InputDevice, Dtypes::Dtype OutputDtype, Shapes::IsShape OutputShape, Devices::Device OutputDevice, const unsigned int threads>
			template<typename InputDtype, typename InputShape, typename InputDevice, typename OutputDtype, typename OutputShape, typename OutputDevice, const unsigned int threads>
			class LayerBox : public Interfaces::ILayer<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, threads> {
			private:
				using InputType = typename InputDtype::Type;
				using OutputType = typename OutputDtype::Type;
			protected:
				
				//protected so that DeviceChange layers and such can access the stores directly, but in general not supposed to be directly accessed
				const Tensors::Tensor<InputDtype, InputShape, threads, InputDevice> inputTensor;
				const Tensors::Tensor < OutputDtype, OutputShape, threads, OutputDevice> outputTensor;


			

			public:

				LayerBox(Tensors::Tensor<InputDtype, InputShape, threads, InputDevice> inputs, Tensors::Tensor<OutputDtype, OutputShape, threads, OutputDevice> outputs) : inputTensor(inputs), outputTensor(outputs) {}

				//thread-specific functions

				void threadForward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					Threads::ThreadContext& threadContext
				) = delete;

				void threadBackward(
						References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
						References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
						References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
						References::MutableShapedDeviceReference<InputType, InputShape, InputDevice> preNodeDeltas,
						Threads::ThreadContext& threadContext
				) = delete;




	

				void threadModifyWeights(
						References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
						References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
						Threads::ThreadContext & threadContext
				) = delete;












				//provide a default implementation for forward here, rather than in the interface, as the interface does not store the input/outputs locations
				template<Layer LayerT>
				static inline void forward(LayerT& layer, Threads::ThreadContext& ctx) {
					ctx.synchronize();
					layer.threadForward(layer.inputTensor.link->data.asImmutable(ctx), layer.outputTensor.link->data.asMutable(ctx), ctx);
				}

				//implement backwards
				template<Layer LayerT>
				static void backward(LayerT& layer, Threads::ThreadContext& ctx) {
					ctx.synchronize();
					layer.threadBackward(layer.inputTensor.link->data.asImmutable(ctx), layer.outputTensor.link->data.asImmutable(ctx), layer.outputTensor.link->deltas.asMutable(ctx), layer.inputTensor.link->deltas.asMutable(ctx), ctx);
				}

				template<Layer LayerT>
				static inline void modifyWeights(LayerT& layer, Threads::ThreadContext& ctx) {

					layer.threadModifyWeights(layer.inputTensor.link->data.asImmutable(ctx), layer.outputTensor.link->deltas.asImmutable(ctx), ctx);

				}
			};
		}
	}
}


	
