#pragma once
#include "./biased_conv_impl.h"

namespace NN {
	namespace Layers {
		/// <summary>
		/// Biased Convoloution Layer
		/// overlays a kernel of weights over a matrix of inputs and element-wise multiplies, then sums the results and adds a bias for each output
		/// </summary>
		/// <typeparam name="Dtype">The datatype</typeparam>
		/// <typeparam name="Device">The Device to run on</typeparam>
		/// <typeparam name="InputShape">The shape of the input</typeparam>
		/// <typeparam name="OutputShape">The shape of the output</typeparam>
		/// <typeparam name="KernelShape">The shape of the kernel</typeparam>
		/// <typeparam name="WeightAdjuster">The method to use to adjust the weights</typeparam>
		/// <typeparam name="threads">The number of threads</typeparam>
		template<typename Dtype, typename Device, typename InputShape, typename KernelShape, typename OutputShape, const unsigned int threads, typename WeightAdjuster = NN::WeightModifiers::Linear<typename Dtype::Type, 0.0001>>
		class BiasedConvoloution : public BiasedConvoloutionImpl<Dtype, Device, InputShape, KernelShape,OutputShape, OutputShape, threads, WeightAdjuster> //define BiasShape as OutputShape
		{

		public:
			static constexpr StringLiteral name = "Biased Convoloution";
			using BiasedConvoloutionImpl<Dtype, Device, InputShape, KernelShape, OutputShape, OutputShape, threads, WeightAdjuster>::BiasedConvoloutionImpl;
		};
	}
}