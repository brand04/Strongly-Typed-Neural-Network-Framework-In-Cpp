#pragma once
#include "../../asserts/activation_function_asserts.h"
#include "../../kernels/activation.cuh"
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
			/// Wraps a layer inheriting from Abstract::LayerBox so that an activation function is run after it
			/// </summary>
			/// <typeparam name="Layer">The layer to wrap</typeparam>
			/// <typeparam name="ActivationFunction">The activation function to apply</typeparam>
			template<typename Layer,  typename ActivationFunction>
			class ActivationWrapper : public Layer {
				//assert that Layer is in fact a Layer
				static_assert(struct_assert<Asserts::AssertLayer<Layer>>, "ActivationWrapper must wrap a Layer, the type supplied was not");
				
				//extract traits
				using InputShape = typename Traits::getLayerTraits<Layer>::InputShape;
				using OutputShape = typename Traits::getLayerTraits<Layer>::OutputShape;
				using InputDtype = typename Traits::getLayerTraits<Layer>::InputDtype;
				using OutputDtype = typename Traits::getLayerTraits<Layer>::OutputDtype;
				using InputDevice = typename Traits::getLayerTraits<Layer>::InputDevice;
				using OutputDevice = typename Traits::getLayerTraits<Layer>::OutputDevice;
				using OutputType = typename Traits::getLayerTraits<Layer>::OutputType;
				using InputType = typename Traits::getLayerTraits<Layer>::InputType;

				static_assert(struct_assert < Asserts::AssertActivationFunction<ActivationFunction, OutputDtype>>, "The Activation Function supplied does not work on the wrapped layer's dtype");

			public:

				static constexpr StringCollection name = StringCollection(Layer::name, " (", ActivationFunction::name, ")"); //adjust the name to include the Activation Function's name

				inline void threadForward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					Threads::ThreadContext& threadContext
				) {
					Layer::template forward<Layer>(*this, threadContext); //run the wrapped layer first - run Layer's forward function (which is likely to be layerbox's which delegates to threadForward of the input type
					//- which unless we aren't careful would be ActivationWrapper<Layer>. To avoid this, enforce the input type to be  Layer so that *this is casted down to Layer type, thus avoiding infinite recursion
					threadContext.synchronize(); //wait for operation to complete
					Kernels::ActivationFunction::forward<ActivationFunction>(outputs, threadContext);

				}

				inline void threadBackward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
					References::MutableShapedDeviceReference < InputType, InputShape, InputDevice> preNodeDeltas,
					Threads::ThreadContext& threadContext
				) {
					threadContext.synchronize();
					Kernels::ActivationFunction::backward<ActivationFunction>(postNodeDeltas, outputs, threadContext);
					threadContext.synchronize();
					Layer::template backward<Layer>(*this, threadContext);
				}


				using Layer::Layer;


			};
		}
	}
}