#pragma once
#include "../shapes/includes.h"
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace NN {
	namespace CudaHelpers {

        /// <summary>
        /// initizes weights using stateCount states and loops to reuse states
        /// <typeparam name="Dtype"> The Datatype of the weights to initialize </typeparam>
        /// <typeparam name="stateCount"> The number of random states initialized</typeparam>
        /// <typeparam name="WeightShape"> The Shape of the weights to initialize </typeparam>
        /// <typeparam name ="BiasShape"> The Shape of the biases to initialize </typeparam>
        /// 
        /// </summary>
      

        template<typename Dtype, size_t stateCount, Shapes::IsShape WeightsShape, Shapes::IsShape BiasShape = Shapes::Shape<0>>
        __global__ void initializeWeights(typename Dtype::Type* weights, curandStateXORWOW_t* states, typename Dtype::Type* biasLocation = nullptr) {
            const int id = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (id < stateCount) {
                size_t excess = (WeightsShape::volume + BiasShape::volume) % stateCount;
                size_t repeats = ((WeightsShape::volume + BiasShape::volume) / stateCount) + (id < excess ? 1 : 0); //figure out how many repeats to do for this thread - need to exactly cover the entirety of Weights and Biases using stateCount threads
                for (size_t i = 0; i < repeats; i++) {
                    if (i * stateCount < WeightsShape::volume) {
                        //weight
                        weights[id + (i * stateCount)] = Dtype::init(curand(states + id));
                    }
                    else {
                        //bias
                        biasLocation[id + (i * stateCount) - WeightsShape::volume] = Dtype::init(curand(states + id));
                    }

                }

            }

        }

        /// <summary>
        /// Initializes random states on cuda
        /// <typeparam name="requiredStates">The number of states requested</typeparam>
        /// </summary>
        template<size_t requiredStates>
        __global__ void initCurand(curandStateXORWOW_t* states, unsigned long long seed) {
            const int id = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (id < requiredStates) {
                curand_init(seed, id, 0, states + id);
            }
        }
	}
}