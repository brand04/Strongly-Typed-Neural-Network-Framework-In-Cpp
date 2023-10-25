#pragma once
#include "../../kernels/softmax.cuh"
#include "../../asserts/layer_asserts.h"
#include "../../helpers/fixed_string.h"
#include "../../helpers/string_collection.h"
#include "../../traits/trait_get.h"
#include "../../references/includes.h"
#include "../../threads/thread_context.h"
namespace NN {
	namespace Layers {
		namespace Phantom {

			/// <summary>
			/// Wraps a layer so that a softmax is run on the outputs
			/// INCOMPATIBLE with ActivationFunctionWrapper as both this and the wrapper modify the outputs, whilst also using the outputs in backpropogation
			/// </summary>
			/// <typeparam name="Layer">The layer to wrap</typeparam>
			template<typename Layer>
			class SoftmaxWrapper : public Layer {
				//assert that Layer is in fact a Layer
				static_assert(struct_assert<Asserts::AssertLayerT<Layer>>, "SoftmaxWrapper must wrap a Layer, the type supplied was not");

				//extract traits
				using InputShape = typename Traits::getLayerTraits<Layer>::InputShape;
				using OutputShape = typename Traits::getLayerTraits<Layer>::OutputShape;
				using InputDtype = typename Traits::getLayerTraits<Layer>::InputDtype;
				using OutputDtype = typename Traits::getLayerTraits<Layer>::OutputDtype;
				using InputDevice = typename Traits::getLayerTraits<Layer>::InputDevice;
				using OutputDevice = typename Traits::getLayerTraits<Layer>::OutputDevice;
				using OutputType = typename Traits::getLayerTraits<Layer>::OutputType;
				using InputType = typename Traits::getLayerTraits<Layer>::InputType;

				public:

				static constexpr StringCollection name = StringCollection(Layer::name, " (Softmax)"); //adjust the name to include the Activation Function's name

				inline void threadForward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					Threads::ThreadContext& threadContext
				) {
					Layer::template forward<Layer>(*this, threadContext); //run the wrapped layer first
					threadContext.synchronize(); //wait for operation to complete
					Kernels::Softmax::softmax<OutputDtype, OutputShape, OutputDevice, true>(outputs, outputs, threadContext);
					threadContext.synchronize();

				}

				inline void threadBackward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
					References::MutableShapedDeviceReference < InputType, InputShape, InputDevice> preNodeDeltas,
					Threads::ThreadContext& threadContext
				) {
					threadContext.synchronize();
					Kernels::Softmax::softmaxDerivative<OutputDtype, OutputShape, OutputDevice>(postNodeDeltas, postNodeDeltas, outputs, threadContext);
					threadContext.synchronize();
					Layer::template backward<Layer>(*this, threadContext);
				}


				using Layer::Layer;


			};
		}
	}
}