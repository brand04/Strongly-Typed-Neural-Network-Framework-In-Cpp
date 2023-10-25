#pragma once

#include "./unsafe/includes.h"
#include "../shapes/includes.h"
#include "../threads/thread_context.h"
#include "../threads/mutable_weight_lock.h"
#include "../threads/immutable_weight_lock.h"
#include "../dtypes/concept.h"
#include "../references/includes.h"

#include <atomic>
namespace NN {
	namespace Storage {

		//Stores weights and potentially biases of specified shapes, implementing as the weights are all stored, and then the biases appended 
		template<typename Dtype, typename MatrixShape, typename Device, unsigned int threads, typename BiasShape = Shapes::zero<1>>
		class WeightStore : public Unsafe::UnsafeStore<typename Dtype::Type, MatrixShape::volume + BiasShape::volume, Device> {
			using dtype = typename Dtype::Type;

			/*bit set :
			bits:
				n: locked-mutably ( 1 -> locked, 0 -> unlocked)
				0 - n-1 : number of immutable-readers


			 */

			
			using CounterT = unsigned char; //can increase to larger types if more threads required


			mutable std::atomic<CounterT> lock = 0; //mutable - not part of logical constness, the counter/lock for mutable and immutable accesses


			static constexpr unsigned int bits = sizeof(CounterT) * 8;
			static constexpr unsigned int shiftToMutableLock = bits - 1;
			static_assert(threads < (1 << (shiftToMutableLock)-1), "Too many threads for WeightStore");

			
			// The various refence types returned by the locks
			
			using WeightMutRef = References::MutableShapedDeviceReference<dtype, MatrixShape, Device>;
			using WeightImmutRef = References::ImmutableShapedDeviceReference<dtype, MatrixShape, Device>;
			using BiasMutRef = References::MutableShapedDeviceReference<dtype, BiasShape, Device>;
			using BiasImmutRef = References::ImmutableShapedDeviceReference<dtype, BiasShape, Device>;

			//Base class
			using Super = Unsafe::UnsafeStore<dtype, MatrixShape::volume + BiasShape::volume, Device>;

			//lock classes
			using MutLock = Threads::MutableWeightLock<CounterT, WeightMutRef, BiasMutRef>; //alias the mutable output type
			using ImmutLock = Threads::ImmutableWeightLock<CounterT, WeightImmutRef, BiasImmutRef>;
		public:
			WeightStore() : Unsafe::UnsafeStore<typename Dtype::Type, MatrixShape::volume + BiasShape::volume, Device>() {}
		
			/// <summary>
			/// Spinlocks the thread until a lock can be aquired (ie guarentees that result.success() == true)
			/// </summary>
			/// <returns> A lock whereby lock.success() == true and the weight/bias fields are filled in</returns>
			inline MutLock getAsMutable() {
				MutLock ret;
				do {
					new(&ret) MutLock(lock, Super::asMutable(),Super::asMutable() + MatrixShape::volume);
				} while (!ret.success());
				return ret;
			}
			/// <summary>
			/// See ::getAsMutable
			/// </summary>
			/// <returns></returns>
			inline ImmutLock getAsImmutable() const {
				ImmutLock ret;
				do {
					new (&ret) ImmutLock(lock,Super::asImmutable(), Super::asImmutable() + MatrixShape::volume);
				} while (!ret.success());
				return ret;
			}

			/// <summary>
			/// Returns a lock that may or may not be valid (meaning that in order to use, lock.success() should be checked for being true)
			/// </summary>
			/// <returns>
			/// A lock, whereby either:
			///		lock.success() == true and weights and biases are valid pointers
			/// or :
			///		lock.success() == false, weight = biases = nullptr
			inline ImmutLock tryGetAsImmutable() const {
				ImmutLock ret = ImmutLock(lock, Super::asImmutable(), Super::asImmutable() + MatrixShape::volume);
				return ret;
			}

			/// <summary>
			/// Returns a lock that may or may not be valid (meaning that in order to use, lock.success() should be checked for being true)
			/// </summary>
			/// <returns>
			/// A lock, whereby either:
			///		lock.success() == true and weights and biases are valid pointers
			/// or :
			///		lock.success() == false, weight = biases = nullptr
			/// </returns>
			inline MutLock tryGetAsMutable() {
				MutLock ret = MutLock(lock, Super::asMutable(),Super::asMutable() + MatrixShape::volume);
				return ret;
			}

			

			//copy all data to another store
			template<Devices::Device target>
			inline void copyAllTo(WeightStore<Dtype, MatrixShape, target,threads, BiasShape>& other, const Threads::ThreadContext& ctx) const {
				if constexpr (Devices::CUDADevice<Device> || Devices::CUDADevice<target>) {
					this->template copyTo<target>(other, 0, (MatrixShape::volume + BiasShape::volume) * sizeof(dtype), ctx.stream);
				}
				else if constexpr (Devices::CPUDevice<Device> && Devices::CPUDevice<target>) {
					this->template copyTo<target>(other, 0, (MatrixShape::volume + BiasShape::volume) * sizeof(dtype));
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Copying to CPU and CUDA store are supported");
				}
				
			}
		};
	}
}