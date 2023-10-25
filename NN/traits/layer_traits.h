#pragma once
#include "./layer_traits_forward_declaration.h"
#include "../asserts/false_assert.h"
namespace NN {
	namespace Traits {


		/*
		A type used to unpack traits from an arbitary Layer Type that inherits from the ILayer interface

		In order for types such as sequences to be implemented cleanly, they should only require the overall types of the layers they wrap.
		However in order to function they must know some basic information about the layer, such as the types/devices/shapes/threads

		This LayerTraits type allows this information to be available whilst also maintaining encapsulation and not exposing it through multiple template aliases that could be hidden by a child object

		*/
	

		//template<Dtypes::Dtype I_Dtype, Shapes::IsShape I_Shape, Devices::Device I_Device, Dtypes::Dtype O_Dtype, Shapes::IsShape O_Shape, Devices::Device O_Device, const unsigned int threads>

		/// <summary>
		/// LayerTraits is a type describing information a layer should display to other types in order for them to function correct - an example is in a type such as  Sequence<Layer0, Layer1>, Sequence itself becomes a Layer which requires inputs that would be valid for Layer0 and produces outputs of the same type as Layer1, but must also ensure that Layer0's outputs are valid as inputs to Layer1
		///
		/// <typeparam name="I_Dtype">The Input Datatype</typeparam>
		/// <typeparam name="O_Dtype">The Output Datatype</typeparam>
		/// <typeparam name="I_Shape">The Input Shape</typeparam> 
		/// <typeparam name="O_Shape">The Output Shape</typeparam>
		/// <typeparam name="I_Device">The Input Device</typeparam>
		/// <typeparam name="O_Device">The Output Device</typeparam>
		/// <typeparam name="threads">The number of threads the layer is configured for</typeparam>
		/// 
		///  </summary>
		template<typename I_Dtype, typename I_Shape, typename I_Device, typename O_Dtype, typename O_Shape, typename O_Device, const unsigned int threads>
		struct LayerTraits {
			using InputShape = I_Shape;
			using InputDevice = I_Device;
			using OutputShape = O_Shape;
			using OutputDevice = O_Device;

			using InputDtype = I_Dtype;
			using InputType = typename I_Dtype::Type;
			using OutputDtype = O_Dtype;
			using OutputType = typename O_Dtype::Type;

			static constexpr const unsigned int nThreads = threads;
		};
		
		
		




	}
}