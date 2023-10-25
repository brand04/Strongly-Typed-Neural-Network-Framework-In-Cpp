#pragma once
#include "./biased_overlay_impl.h"
#include "../../functions/weight_modifiers/linear_modifier.h"

namespace NN {
	namespace Layers {
		/// <summary>
		/// Biased Overlay layer - overlays the inputs on a weight matrix, performing element-wise multiplication, before computing the sum of these results and adding a unique bias for each output position
		/// </summary>
		/// <typeparam name="Dtype">The datatype</typeparam>
		/// <typeparam name="Device">The device to run on</typeparam>
		/// <typeparam name="InputShape">The shape of the input</typeparam>
		/// <typeparam name="OutputShape">The shape of the output</typeparam>
		/// <typeparam name="WeightAdjuster">The method to adjust weights</typeparam>
		/// <typeparam name="threads">The number of threads to run on</typeparam>
		template<typename Dtype, typename Device, typename InputShape, typename OutputShape, const unsigned int threads, typename WeightAdjuster = NN::WeightModifiers::Linear<typename Dtype::Type>>
		class BiasedOverlay : public BiasedOverlayImpl<Dtype, Device, InputShape, OutputShape, OutputShape, threads> //define BiasShape as OutputShape
		{
			
		public:
			static constexpr StringLiteral name = "Biased Overlay";
			using BiasedOverlayImpl<Dtype, Device, InputShape, OutputShape, OutputShape, threads>::BiasedOverlayImpl;
		};
	}
}