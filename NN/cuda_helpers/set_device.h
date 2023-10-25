#pragma once
#include "../devices/includes.h"
#include <iostream>
#include <cuda_runtime.h>
namespace NN {
	namespace CudaHelpers {

		/// <summary>
		/// Sets the current cuda device, or throws a cudaError_t
		/// </summary>
		/// <typeparam name="Device">The CUDA device to set to</typeparam>
		template<Devices::CUDADevice Device>
		inline void setDevice() {
			cudaError_t err = cudaSetDevice(Device::deviceId);
			if (err != cudaSuccess) {
				std::cerr << "Error setting CUDA device \n";
				throw err;
			}
		}
	}
}