#pragma once
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "../threads/thread_context.h"
#include <iostream>
#include "./rand_kernels.cuh"
#include "../cuda_helpers/kernel_launch.h"
#include "../devices/device.h"

namespace NN {
	namespace CudaHelpers {


		/// <summary>
		/// Sets up random seeds for random weight initialization on a cuda device, requires a call to freeCurand after
		/// </summary>
		/// <typeparam name="Device">The CUDA device</typeparam>
		/// <typeparam name="requiredStates">The number required random states</typeparam>
		/// <param name="ctx">A thread context</param>
		/// <returns></returns>
		template<Devices::CUDADevice Device, size_t requiredStates>
		static __host__ curandStateXORWOW_t* initCurand(Threads::ThreadContext& ctx) {
			static_assert(requiredStates > 0, "Must initialize at least 1 block");
			cudaError_t error = cudaSetDevice(Device::deviceId);

			if (error != cudaSuccess) {
				std::cerr << "Error during random initialization - could not set device - " << cudaGetErrorString(error) << " : " << cudaGetErrorName(error) << "\n";
				throw error;
			}

			curandStateXORWOW_t* states;
			error = cudaMallocAsync(&states, sizeof(curandStateXORWOW_t)* requiredStates, ctx.stream);
			if (error != cudaSuccess) {
				std::cerr << "Error allocating memory for random seed " << cudaGetErrorString(error) << " : " << cudaGetErrorName(error) << "\n";
				throw error;
			}

			ctx.synchronize();
		
			initCurand<requiredStates> << < CudaHelpers::computeGridSize(requiredStates), CudaHelpers::computeBlockSize(requiredStates), 0, ctx.stream >> > (states, (~ctx.syncEpoch) ^ time(NULL));
			ctx.synchronize();
			return states;
		}

		/// <summary>
		/// Frees the memory used for random intialization
		/// </summary>
		/// <param name="states">A pointer to the random states</param>
		/// <param name="ctx">A thread context</param>
		/// <returns></returns>
		static __host__ const void freeCurand(curandStateXORWOW_t* states, Threads::ThreadContext& ctx) {
			ctx.synchronize();
			cudaError_t error = cudaFree(states);
			if (error != cudaSuccess) {
				std::cerr << "Error freeing cuda states for randomization\n";
			}
		}
	}
}