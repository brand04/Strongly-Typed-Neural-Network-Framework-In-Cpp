#pragma once
#include <atomic>

namespace NN {
	namespace Threads {


		/// <summary>
		/// An attempted mutable lock on a layer's weights
		/// 
		/// if this->success(), the mutable lock was aquired successfully
		/// else is was not
		/// </summary>
		/// <typeparam name="CounterT">The type of the counter</typeparam>
		/// <typeparam name="WeightRefT"> The type of the reference to weights provided</typeparam>
		/// <typeparam name="BiasRefT"> The type of the reference to biases provided</typeparam>
		template<typename CounterT, typename WeightRefT, typename BiasRefT>
		class MutableWeightLock;
		
		template<typename CounterT, typename Weightdtype, typename WeightShape, typename WeightDevice, typename Biasdtype, typename BiasShape, typename BiasDevice>
		class MutableWeightLock<CounterT, References::MutableShapedDeviceReference<Weightdtype, WeightShape, WeightDevice>, References::MutableShapedDeviceReference<Biasdtype, BiasShape, BiasDevice>> {
		private:
			static constexpr unsigned int lockIndex = (sizeof(CounterT) * 8) - 1;
			std::atomic<CounterT>* lock = nullptr;
			using WeightRefT = References::MutableShapedDeviceReference<Weightdtype, WeightShape, WeightDevice>;
			using BiasRefT = References::MutableShapedDeviceReference<Biasdtype, BiasShape, BiasDevice>;
			Weightdtype* weights = nullptr;
			Biasdtype* biases = nullptr;
		public:

			inline WeightRefT getWeights() {
				return WeightRefT(weights);
			}
			inline BiasRefT getBiases() {
				return BiasRefT(biases);
			}

			MutableWeightLock(std::atomic<CounterT>& mutableLock, Weightdtype* weights, Biasdtype* biases) : lock(&mutableLock) {
				if ((((*lock).fetch_or(1 << lockIndex) >> lockIndex) & 1) == 0) { //aquire lock
					this->weights = weights;
					this->biases = biases;
				}
			}
			MutableWeightLock() {};

			inline bool success() {
				return weights != nullptr; //biases may be nullptr if no biases stored, but there should always be a weight so use this to determine if we aquired the lock
			}

			//remove lock when falling out of scope
			~MutableWeightLock()
			{
				if (success()) (*lock).fetch_and(~(1 << lockIndex)); //remove lock if successful
			}
		};

	}
}