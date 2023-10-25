#pragma once
#include "../../references/includes.h"
#include "../../threads/thread_context.h"
#include "./forward_declarations.h"
#include "../../traits/layer_traits.h"
#include "../../traits/trait_get.h"

#include "../../helpers/string_collection.h"

#include "../../asserts/type_asserts.h"


#include "../../concepts/is_string_t_concept.h"

namespace NN {
	namespace Layers {
		namespace Interfaces {

			/**
			The base class of all layers.

			Describes a transformation from a tensor of type U, shape InputShape, stored on device InputDevice\n
			to a tensor of type V, shape OutputShape, Stored on Device OutputDevice \n
			that operates with a number of threads in parallel

			All required functions are non-virtual (since everything can be deduced at compile-time anyway), and marked as deleted rather than undefined, so that the compiler error is tracked rather than "unresolved external"

			*/
			template<typename InputDtype, typename InputShape, typename InputDevice, typename OutputDtype, typename OutputShape, typename OutputDevice, const unsigned int threads> //prefer typename-only to concepts so that we can static_assert with structs for errors
			//template<Dtypes::Dtype U, Shapes::IsShape InputShape, Devices::Device InputDevice, Dtypes::Dtype V, Shapes::IsShape OutputShape, Devices::Device OutputDevice, const unsigned int threads> - constraint definition
			class ILayer {
				static_assert(struct_assert<Asserts::AssertLayerParameters<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice>>,"Layer parameters malformed");  //compile-time error message if wrong types

			
				//get the wrapped types of the datatype
				using InputType = typename InputDtype::Type;
				using OutputType = typename OutputDtype::Type;
				//defines for layer operations
			
			public:
				using LayerTraits = Traits::LayerTraits<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, threads>;
				//enforce compile-time evaluation
				static constexpr StringLiteral name = StringLiteral("Layer Interface");

				//sort-of similar to RCTP in the fact that we ask a subclass for information from a base class
				//define a default pattern for construction of fullname from a LayerT::name and traits




				template<typename LayerT, typename prepend = StringT<>, typename prepender = StringT<'\t'>> //default to no prepend value
				static constexpr const StringCollection fullname = StringCollection(prepend::string , LayerT::name, " (" , Traits::getLayerTraits<LayerT>::InputShape::string , " [ " , InputDevice::string , " ] -> ", Traits::getLayerTraits<LayerT>::OutputShape::string , " [ " , OutputDevice::string , " ] )");


			public:

				/*
				
				Thread-specific forwards pass function prototype if using Abstract::LayerBox
				
				void threadForward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					Threads::ThreadContext& threadContext
				) = delete;

				
				Thread-specific backwards pass function prototype if using Abstract::LayerBox
				
				void threadBackward(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
					References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
					References::MutableShapedDeviceReference<InputType, InputShape, InputDevice> preNodeDeltas,
					Threads::ThreadContext& threadContext
				) = delete;



				
				Thread-specific modification of the layer's weights if using Abstract::LayerBox
				Overriden or compile-time error
				
				void threadModifyWeights(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
					Threads::ThreadContext& threadContext
				) = delete;

				*/

				//use static templated functions rather than member functions so that the correct threadForward is run, as these are overriden but not virtual

				template<typename LayerT>
				static void forward(LayerT& layer, Threads::ThreadContext& ctx) = delete; //requires implementation of forward

				template<typename LayerT>
				static void backward(LayerT& layer, Threads::ThreadContext& ctx) = delete; //requires implementation of backward

				template<typename LayerT>
				static void modifyWeights(LayerT& layer, Threads::ThreadContext& ctx) = delete; //requires implementation of modifyWeights

				template<typename LayerT>
				static void saveAsBytes(LayerT& layer, std::ofstream& file, Threads::ThreadContext& ctx) = delete; //requires implementation of saveAsBytes to use

				template<typename LayerT>
				static void readFromBytes(LayerT& layer, std::ifstream& file, Threads::ThreadContext& ctx) = delete; //requires implementation of readFromBytes to use

				/*
				Output the weights to the stringstream, with each line prepended by prepend.
				*/
				void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) = delete;


				/*
				Setup the weights for the first time to some default/random values
				*/
				void initializeWeights(Threads::ThreadContext& threadContext) = delete;


				/*
				An unsafe way to specify the weights for a network
				The pointer should contain the weights, in sequence (along forward direction), as bytes (with varying lengths potentially)
				*/
				void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext) = delete;
			};
		}
	}
}