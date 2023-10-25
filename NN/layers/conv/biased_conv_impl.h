#pragma once

#include <fstream>




#include "../../references/includes.h"
#include "../../shapes/includes.h"
#include "../../helpers/array_to_string.h"



#include "../../asserts/false_assert.h"

#include "../../helpers/fixed_string.h"

#include "../../cuda_helpers/rand.h"
#include "../../cuda_helpers/kernel_launch.h"


#include "../../kernels/convoloution.cuh"
#include "../../kernels/initialization.cuh"
#include "../../kernels/copy.cuh"

#include "../../functions/weight_modifiers/linear_modifier.h"
#include "../../functions/weight_modifiers/clamped_linear_modifier.h"
#include "../../functions/weight_modifiers/clamped_logistic_modifier.h"
#include "../../functions/weight_modifiers/decay_clamped_linear_modifier.h"#
#include "../../functions/weight_modifiers/clamped_offset_modifier.h"

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
		/// Convoloution Layer Implementation
		/// Given a tensor of shape InputShape, and weights of shape KernelShape, produce an output of shape OutputShape through overlaying the kernel over the input and performing element-wise multiplication and then summating the results.
		/// Note that for any given combination of InputShape and KernelShape, there is only a single valid OutputShape, such that InputShape - KernelShape + unit = OutputShape
		/// The Bias should either be Shape<0> or OutputShape
		/// </summary>
		/// <typeparam name="Dtype"> The datatype - which describes the underlying type of the inputs/outputs/weights, the additive/multiplicative identity as well as how to inialize the weights</typeparam>
		/// <typeparam name="Device">The device</typeparam>
		/// <typeparam name="OutputShape">The shape of the output</typeparam>
		/// <typeparam name="KernelShape">The shape of the input</typeparam>
		/// <typeparam name="BiasShape">The shape of the bias - either Shape<0> or OutputShape</typeparam>
		/// <typeparam name="InputShape">The shape of the Input</typeparam>
		/// <typeparam name="WeightAdjuster">The method of Adjusting the weights : TODO: allow non-static storage</typeparam>
		/// <typeparam name="threads">The number of threads to operate on</typeparam>
		template<typename Dtype, typename Device, typename InputShape, typename KernelShape, typename OutputShape, typename BiasShape, const unsigned int threads, typename WeightAdjuster = NN::WeightModifiers::Linear<typename Dtype::Type>>
		class BiasedConvoloutionImpl : public Abstract::LayerBox<Dtype, InputShape, Device, Dtype, OutputShape, Device, threads> {
			

			static_assert(std::same_as<typename KernelShape::add<OutputShape>::subtract<Shapes::unit<KernelShape::dimension>>, InputShape>, "Shape Mismatch - the InputShape should be equal to KernelShape+OutputShape-unit");
			using dtype = typename Dtype::Type; // alias the underlying type

		protected:
			Storage::WeightStore<Dtype, KernelShape, Device, threads, BiasShape> weights; //Create a store of weights on the device
		public:
			static constexpr StringLiteral name = StringLiteral("Biased Convoloution Implementation"); // override the name inherited from LayerBox

			/// <summary>
			/// A function called by the superclass, which provides a thread-specific input and output to read/modify
			/// </summary>
			/// <param name="input"> An immutable block of memory containing the inputs for this layer and thread</param>
			/// <param name="output"> An immutable block of memory containing the inputs for the next layer and this thread, which should be overwritten</param>
			/// <param name="threadContext">The context of this thread</param>
			inline void threadForward(
				References::ImmutableShapedDeviceReference<dtype, InputShape, Device> input,
				References::MutableShapedDeviceReference<dtype, OutputShape, Device> output,
				Threads::ThreadContext& threadContext
			) {
				auto lock = weights.getAsImmutable(); //gets a read lock on the weights - polling repeatedly until available
				Kernels::Convoloution::convoloution<Dtype>(output, input, lock.getWeights(), lock.getBiases(), threadContext); // Run the kernel on the correct device
				threadContext.synchronize(); //synchronize this thread
			}

			/// <summary>
			/// A function called by the superclass, which provides the thread-specific inputs and outputs for this layer and thread, as well as the partial derivative of the error W.R.T the outputs (in postNodeDeltas), for which the partial derivative of the error W.R.T the inputs should be computed
			/// (and stored within preNodeDeltas). Note that the postNodeDeltas are also mutable - to allow any subclasses to modify them
			/// </summary>
			/// <param name="inputs">The inputs that were given to this layer and thread</param>
			/// <param name="outputs">The outputs produced by this layer and thread (including any wrappers)</param>
			/// <param name="postNodeDeltas">The node deltas for the outputs (after processing by any wrappers)</param>
			/// <param name="preNodeDeltas"> The node deltas to be computed</param>
			/// <param name="threadContext">The context for this thread</param>
			void threadBackward(
				References::ImmutableShapedDeviceReference<dtype, InputShape, Device> inputs,
				References::ImmutableShapedDeviceReference<dtype, OutputShape, Device> outputs,
				References::MutableShapedDeviceReference<dtype, OutputShape, Device> postNodeDeltas,
				References::MutableShapedDeviceReference<dtype, InputShape, Device> preNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				auto lock = weights.getAsImmutable(); //obtain read lock
				Kernels::Convoloution::backpropMatrixDeltas<Dtype,KernelShape, InputShape, OutputShape, Device>(preNodeDeltas, postNodeDeltas, lock.getWeights(), threadContext); //backpropogate on the correct device
				threadContext.synchronize(); //synchronize
			}

			/// <summary>
			/// A function called by the superclass, which provides the thread-specific inputs and output-deltas for this layer and thread, to be used in the modification of weights
			/// </summary>
			/// <param name="inputs">The inputs supplied to this layer</param>
			/// <param name="postNodeDeltas">The partial derivatives of the error W.R.T the outputs</param>
			/// <param name="threadContext">The context for this thread</param>
			void threadModifyWeights(
				References::ImmutableShapedDeviceReference<dtype, InputShape, Device> inputs,
				References::ImmutableShapedDeviceReference<dtype, OutputShape, Device> postNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadContext.synchronize();
				auto lock = weights.getAsMutable(); //obtain mutable lock (we intend to mutate the weights)
			
				Kernels::Convoloution::adjustKernelWeights<Dtype, WeightAdjuster>(lock.getWeights(), lock.getBiases(), inputs, postNodeDeltas, threadContext);
				threadContext.synchronize();
			}

			/// <summary>
			/// Save the weights as bytes to a file
			/// </summary>
			/// <typeparam name="LayerT"></typeparam>
			/// <param name="layer"></param>
			/// <param name="file"></param>
			/// <param name="ctx"></param>
			template<typename LayerT>
			static void saveAsBytes(LayerT& layer, std::ofstream& file, Threads::ThreadContext& ctx) {

				Storage::WeightStore<Dtype, KernelShape, Devices::CPU, threads, BiasShape> cpuWeights;
				layer.weights.copyAllTo(cpuWeights, ctx);
				ctx.synchronize();
				auto lock = cpuWeights.getAsImmutable();
				file.write(reinterpret_cast<const char*>(lock.getWeights().ptr), sizeof(dtype) * (KernelShape::volume + BiasShape::volume));
			}

			template<typename LayerT>
			static void readFromBytes(LayerT& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
				if constexpr (!Devices::CPUDevice<Device>) {
					Storage::WeightStore<Dtype, KernelShape, Devices::CPU, threads, BiasShape> cpuWeights;
					auto lock = cpuWeights.getAsMutable();
					file.read(reinterpret_cast<char*>(lock.getWeights().ptr), sizeof(dtype) * (KernelShape::volume + BiasShape::volume));
					cpuWeights.copyAllTo(layer.weights, ctx);
					ctx.synchronize();
				}
				else {
					auto lock = layer.weights.getAsMutable();
					file.read(reinterpret_cast<char*>(lock.getWeights().ptr), sizeof(dtype) * (KernelShape::volume + BiasShape::volume));
				}
			}

			//displaying weights on cuda is more difficult - the solution gone for here is the temporary allocation and copying of weights to ram
			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) const {
				Storage::WeightStore<Dtype, KernelShape, Devices::CPU, threads, BiasShape> tmp;
				weights.template copyAllTo<Devices::CPU>(tmp, threadContext);
				threadContext.synchronize(); //wait for copy to complete
				auto lock = tmp.getAsImmutable();
				Helpers::arrayToString<dtype, KernelShape>(s, prepend, lock.getWeights());
				if constexpr (BiasShape::volume > 0) {
					s << prepend << "Biases: ";
					Helpers::arrayToString<dtype, BiasShape>(s, prepend, lock.getBiases());
				}

				//tmp goes out of scope and is freed
			}
			/// <summary>
			/// Initialize the weights according to the method specified by the datatype
			/// </summary>
			/// <param name="threadContext">The context of this thread</param>
			inline void initializeWeights(Threads::ThreadContext& threadContext) {
				auto lock = weights.getAsMutable(); //obtain mutable lock
				Kernels::Initialization::intialize<Dtype>(lock.getWeights(), lock.getBiases(), threadContext);

			}


			/*
			An unsafe way to specify the weights for a network
			The pointer should contain the weights, in sequence (along forward direction), as bytes (with varying lengths potentially)
			*/
			void setWeights(uint8_t*& handle, const Threads::ThreadContext& threadContext) {
				auto lock = weights.getAsMutable();
				Kernels::Copy::copy<KernelShape::volume>(handle, lock.getWeights(), threadContext);
				Kernels::Copy::copy<BiasShape::volume>(handle, lock.getBiases(), threadContext);
				handle += sizeof(dtype) * (KernelShape::volume + BiasShape::volume); //increment the handle to the next layer's weights
			}

			using Abstract::LayerBox<Dtype, InputShape, Device, Dtype, OutputShape, Device, threads>::LayerBox;

		};
	}
}