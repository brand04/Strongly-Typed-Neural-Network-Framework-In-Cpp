#pragma once
#include <atomic>

namespace NN {
	namespace Threads {



		/// <summary>
		/// An attempted immutable lock on a layer's weights
		/// 
		/// if this->success(), the mutable lock was aquired successfully
		/// else is was not
		/// </summary>
		/// <typeparam name="CounterT">The type of the counter</typeparam>
		/// <typeparam name="WeightRefT"> The type of the reference to weights provided</typeparam>
		/// <typeparam name="BiasRefT"> The type of the reference to biases provided</typeparam>
		template<typename CounterT, typename WeightRefT, typename BiasRefT>
		class ImmutableWeightLock;

		template<typename CounterT, typename Weightdtype, typename WeightShape, typename WeightDevice, typename Biasdtype, typename BiasShape, typename BiasDevice>
		class ImmutableWeightLock<CounterT, References::ImmutableShapedDeviceReference<Weightdtype,WeightShape,WeightDevice>, References::ImmutableShapedDeviceReference<Biasdtype, BiasShape, BiasDevice>> {
		private:
			static constexpr unsigned int lockIndex = (sizeof(CounterT) * 8) - 1;
			std::atomic<CounterT>* lock = nullptr;
			Weightdtype const * weights = nullptr;
			Biasdtype const* biases = nullptr;

			using BiasRefT = References::ImmutableShapedDeviceReference<Biasdtype, BiasShape, BiasDevice>;
			using WeightRefT = References::ImmutableShapedDeviceReference<Weightdtype, WeightShape, WeightDevice>;
		public:
			inline BiasRefT getBiases() {
				return BiasRefT(biases);
			}
			inline WeightRefT getWeights() {
				return WeightRefT(weights);
			}
			ImmutableWeightLock(std::atomic<CounterT>& mutableLock, Weightdtype const* const weights, Biasdtype const* const biases) : lock(&mutableLock) {
				if ((*lock)++ == 0){ //aquire lock TODO:allow multiple read-locks
					this->weights = weights;
					this->biases = biases;
				}
				else {
					(*lock)--;
				}
			}
			ImmutableWeightLock() {};
			
		
			inline bool success() {
				return weights != nullptr;
			}

			//decerement counter once falls out of scope to free this specific immutable lock (readlock)
			~ImmutableWeightLock()
			{
				if (success()) (*lock)--;
			}
		};

	}
}