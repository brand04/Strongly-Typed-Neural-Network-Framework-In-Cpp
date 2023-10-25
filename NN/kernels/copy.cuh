#pragma once
#include "../devices/includes.h"
#include "../threads/thread_context.h"
#include "../references/includes.h"#
#include "../shapes/shape_concept.h"
#include "../asserts/false_assert.h"
#include <stdio.h>
#include <cuda_runtime.h>
namespace NN {
	namespace Kernels {
		namespace Copy {

			template<typename T, Shapes::IsShape Shape, Devices::Device Device>
			inline void copy(void const* const src, References::MutableShapedDeviceReference<T,Shape,Device> dst, Threads::ThreadContext& ctx) {
				if constexpr (Devices::CPUDevice<Device>) {
					memcpy(dst, src, sizeof(T)*Shape::volume);
				}
				else if constexpr (Devices::CUDADevice<Device>) {
					cudaError_t error;
					error = cudaSetDevice(Device::deviceId);
					if (error != cudaSuccess) {
						std::cerr << "Error setting device during copying - " << cudaGetErrorName(error) << " : " << cudaGetErrorString(error) << "\n";
						throw error;
					}

					error = cudaMemcpy(dst, src, sizeof(T)*Shape::volume, cudaMemcpyHostToDevice);
				}
				else {
					//make dependant so that the assert is not always falsified
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CUDA and CPU are supported");
				}
			}
		}
	}
}