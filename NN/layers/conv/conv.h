#pragma once
#include "./biased_conv_impl.h"

namespace NN {
	namespace Layers {
		/// <summary>
		/// Unbiased Convoloution layer - overlays a kernel of weights onto an input matrix, element-wise multiplying and then sums the results
		/// </summary>
		/// <typeparam name="Dtype">The Datatype</typeparam>
		/// <typeparam name="Device">The Device to run on</typeparam>
		/// <typeparam name="InputShape">The shape of the input</typeparam>
		/// <typeparam name="KernelShape">The shape of the kernel</typeparam>
		/// <typeparam name="OutputShape">The shape of the output</typeparam>
		/// <typeparam name="Adjuster">The method of adjusting the weights</typeparam>
		/// <typeparam name="threads">The number of threads to run on</typeparam>
		template<typename Dtype, typename Device, typename InputShape, typename KernelShape, typename OutputShape, const unsigned int threads, typename Adjuster = NN::WeightModifiers::Linear<typename Dtype::Type>>
		class Convoloution : public BiasedConvoloutionImpl<Dtype, Device, InputShape, KernelShape, OutputShape, Shapes::Shape<0>, threads> //define BiasShape as 0
		{

		public:
			static constexpr StringLiteral name = "Unbiased Convoloution";
			using BiasedConvoloutionImpl<Dtype, Device, InputShape, KernelShape, OutputShape, Shapes::Shape<0>, threads>::BiasedConvoloutionImpl;
		};
	}
}