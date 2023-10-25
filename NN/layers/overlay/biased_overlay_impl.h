#pragma once
#include <fstream>

#include "../../references/includes.h"
#include "../../shapes/includes.h"
#include "../../helpers/array_to_string.h"

#include "../../asserts/false_assert.h"

#include "../../helpers/fixed_string.h"

#include "../../cuda_helpers/rand.h"
#include "../../cuda_helpers/kernel_launch.h"



#include "../../kernels/overlay.cuh"
#include "../../kernels/initialization.cuh"
#include "../../kernels/copy.cuh"

#include "../../functions/weight_modifiers/linear_modifier.h"

#include "../../functions/weight_modifiers/no_modification.h"

#include "../../storage/weight_store.h"

#include "../abstract/layer_box.h"

#include <curand.h>

namespace NN {
	namespace Layers {
	


		//Conceptual template declaration
		//template<Dtypes::Dtype Dtype, Shapes::IsShape InputShape, Shapes::IsShape OutputShape, const unsigned int threads>

		//changed to purely typename with struct assertions because concept error messages on large types can be unreadable, whereas the struct_assert breaks it down into smaller and smaller components

		/// <summary>
		/// An Overlay Layer: A layer which overlays the input onto a weight matrix and performs elemental-wise multiplication, then sums the results and possibly adds a bias
		/// </summary>
		/// <typeparam name="Dtype">The datatype of the weights/inputs/outputs</typeparam>
		/// <typeparam name="Device">The device to run on</typeparam>
		/// <typeparam name="InputShape">The shape of the input</typeparam>
		/// <typeparam name="OutputShape">The shape of the output (which when combined with the shape of the input can be used to infer the shape of the weight matrix required</typeparam>
		/// <typeparam name="BiasShape">The shape of the bias : should be Shape<0> or OutputShape</typeparam>
		/// <typeparam name="WeightAdjuster">The method of adjsuting the weights</typeparam>
		/// <typeparam name="threads"The number of threads to run on></typeparam>
		template<typename Dtype, typename Device, typename InputShape, typename OutputShape, typename BiasShape, const unsigned int threads, typename WeightAdjuster = NN::WeightModifiers::Linear<typename Dtype::Type>>
		class BiasedOverlayImpl : public Abstract::LayerBox<Dtype, Shapes::PadToMax<InputShape, OutputShape>, Device, Dtype, Shapes::PadToMax<OutputShape, InputShape>, Device, threads> {
			
			//pad to max dimension 
			using InputShapePadded = Shapes::PadToMax<InputShape, OutputShape>;
			using OutputShapePadded = Shapes::PadToMax<OutputShape, InputShape>;


			using MatrixShape = InputShapePadded::add<OutputShapePadded>::subtract<Shapes::unit<InputShapePadded::dimension>>; //infer weight shape
			using dtype = typename Dtype::Type; //alias underlying type

		protected:
			Storage::WeightStore<Dtype, MatrixShape, Device, threads, BiasShape> weights;
		public:
			static constexpr StringLiteral name = StringLiteral("Biased Overlay Implementation");

			/// <summary>
			/// Called by the superclass, which provides a reference to a block of memory where the inputs for this layer and thread are stored, as well as the next layer's inputs (to output to)
			/// </summary>
			/// <param name="inputs">The inputs to the layer</param>
			/// <param name="outputs">The location to store the outputs of the layer</param>
			/// <param name="threadContext">The context of the thread</param>
			void threadForward(
				References::ImmutableShapedDeviceReference<dtype, InputShapePadded, Device> inputs,
				References::MutableShapedDeviceReference<dtype, OutputShapePadded, Device> outputs,
				Threads::ThreadContext& threadContext
			) {

				auto lock = weights.getAsImmutable(); //obtain read lock
				Kernels::Overlay::slidingKernel<Dtype>(outputs, inputs, lock.getWeights(), lock.getBiases(), threadContext);
				threadContext.synchronize();
			}

			/// <summary>
			/// Called by the superclass, which provides a reference to the same block of memory as used in threadForward for storing inputs/outputs (this time as immutable blocks), as well as the node deltas computed (postNodeDeltas) and node deltas to compute in this pass via backpropogation (preNodeDeltas)
			/// </summary>
			/// <param name="inputs">The inputs that were given to this layer</param>
			/// <param name="outputs">The outputs produced by this layer</param>
			/// <param name="postNodeDeltas">The partial derivatives of the error W.R.T this layer's outputs</param>
			/// <param name="preNodeDeltas">The partial derivative of the error W.R.T this layer's inputs - which should be computed here</param>
			/// <param name="threadContext">The context for the thread</param>
			void threadBackward(
				References::ImmutableShapedDeviceReference<dtype,InputShapePadded, Device> inputs,
				References::ImmutableShapedDeviceReference<dtype, OutputShapePadded, Device> outputs,
				References::MutableShapedDeviceReference<dtype, OutputShapePadded, Device> postNodeDeltas,
				References::MutableShapedDeviceReference<dtype, InputShapePadded, Device> preNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				auto lock = weights.getAsImmutable();
				Kernels::Overlay::backpropDeltasKernel<Dtype,Device,InputShapePadded,OutputShapePadded,MatrixShape>(preNodeDeltas, postNodeDeltas, lock.getWeights(), threadContext);
				threadContext.synchronize();
			}

			/// <summary>
			/// Called by the superclass, which provides a reference to this layer's inputs, and the partial derivatives of the error W.R.T this layer's outputs, from which the weights should be modified according to their individual partial derivatives computed via the chain rule
			/// </summary>
			/// <param name="inputs">The inputs to this layer</param>
			/// <param name="postNodeDeltas">The node deltas for the outputs</param>
			/// <param name="threadContext">The context for this thread</param>
			void threadModifyWeights(
				References::ImmutableShapedDeviceReference<dtype, InputShapePadded, Device> inputs,
				References::ImmutableShapedDeviceReference<dtype, OutputShapePadded, Device> postNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				
				threadContext.synchronize();
				auto lock = weights.getAsMutable();
				Kernels::Overlay::adjustWeightsKernel<Dtype, WeightAdjuster>(postNodeDeltas, inputs, lock.getWeights(), lock.getBiases(), threadContext);
				threadContext.synchronize();
			}

			template<typename LayerT>
			static void saveAsBytes(LayerT& layer, std::ofstream& file, Threads::ThreadContext& ctx) {

				Storage::WeightStore<Dtype, MatrixShape, Devices::CPU, threads, BiasShape> cpuWeights;
				layer.weights.copyAllTo(cpuWeights,ctx);
				ctx.synchronize();
				auto lock = cpuWeights.getAsImmutable();
				file.write(reinterpret_cast<const char*>(lock.getWeights().ptr), sizeof(dtype) * (MatrixShape::volume + BiasShape::volume));
			}

			template<typename LayerT>
			static void readFromBytes(LayerT& layer, std::ifstream& file, Threads::ThreadContext& ctx) {

				Storage::WeightStore<Dtype, MatrixShape, Devices::CPU, threads, BiasShape> cpuWeights;
				auto lock = cpuWeights.getAsMutable();
				file.read(reinterpret_cast<char*>(lock.getWeights().ptr), sizeof(dtype) * (MatrixShape::volume + BiasShape::volume));
				cpuWeights.copyAllTo(layer.weights, ctx);
				ctx.synchronize();
			}

			//displaying weights on cuda is more difficult - the solution gone for here is the temporary allocation and copying of weights to CPU ram
			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) const {
				Storage::WeightStore<Dtype, MatrixShape, Devices::CPU, threads, BiasShape> tmp;
				weights.template copyAllTo<Devices::CPU>(tmp,threadContext);
				threadContext.synchronize(); //wait for copy to complete
				auto lock = tmp.getAsImmutable();
				Helpers::arrayToString<dtype, MatrixShape>(s, prepend, lock.getWeights());
				if constexpr (BiasShape::volume > 0) {
					s << prepend <<"Biases: ";
					Helpers::arrayToString<dtype, BiasShape>(s, prepend, lock.getBiases());
				}

				//tmp goes out of scope and is freed
			}

			void initializeWeights(Threads::ThreadContext& threadContext) {
				auto lock = weights.getAsMutable();
				Kernels::Initialization::intialize<Dtype>(lock.getWeights(), lock.getBiases(), threadContext);
				
			}


			/*
			An unsafe way to specify the weights for a network
			The pointer should contain the weights, in sequence (along forward direction), as bytes (with varying lengths potentially)
			*/
			void setWeights(uint8_t*& handle, const Threads::ThreadContext& threadContext) {
				
				Kernels::Copy::copy<MatrixShape::volume>(handle, weights.getWeightsAsMutable(), threadContext);
				Kernels::Copy::copy<BiasShape::volume>(handle, weights.getBiasesAsMutable(), threadContext);
				handle += sizeof(dtype) * (MatrixShape::volume + BiasShape::volume); //increment the handle to the next layer's weights
			}

			using Abstract::LayerBox<Dtype, InputShapePadded, Device, Dtype, OutputShapePadded, Device, threads>::LayerBox;

		};
	}
}