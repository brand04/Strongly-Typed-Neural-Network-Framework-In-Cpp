#pragma once
#include "../abstract/unweighted.h"
#include "../../kernels/softmax.cuh"

namespace NN {
	namespace Layers {
		/// <summary>
		/// Computes the stable softmax for a given input
		/// Softmax is defined as e^x / the sum of e^x for all xs
		/// </summary>
		/// <typeparam name="Dtype">The type of the data (should be floating)</typeparam>
		/// <typeparam name="Shape">The shape of the input and output</typeparam>
		/// <typeparam name="Device">The device to run on</typeparam>
		/// <typeparam name="threads">The number of threads to handle</typeparam>
		template<typename Dtype, typename Shape, typename Device, unsigned int  threads>
		class Softmax : public Abstract::UnweightedLayer<Dtype, Shape, Device, Dtype, Shape, Device, threads> {
		private:
			using dtype = typename Dtype::Type;
		public:
			static constexpr StringLiteral name = StringLiteral("Softmax");
			inline void threadForward(
				References::ImmutableShapedDeviceReference<dtype, Shape, Device> inputs,
				References::MutableShapedDeviceReference<dtype, Shape, Device> outputs,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				Kernels::Softmax::softmax<Dtype, Shape, Device, true>(inputs, outputs, threadContext);
				threadContext.synchronize();
			}

			inline void threadBackward(
				References::ImmutableShapedDeviceReference<dtype, Shape, Device> inputs,
				References::ImmutableShapedDeviceReference<dtype, Shape, Device> outputs,
				References::MutableShapedDeviceReference<dtype, Shape, Device> postNodeDeltas,
				References::MutableShapedDeviceReference<dtype, Shape, Device> preNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				Kernels::Softmax::softmaxDerivative<Dtype, Shape, Device>(preNodeDeltas, postNodeDeltas, outputs, threadContext);
				threadContext.synchronize();
			}

			using Abstract::UnweightedLayer<Dtype, Shape, Device, Dtype, Shape, Device, threads>::UnweightedLayer;



		};
	}
}