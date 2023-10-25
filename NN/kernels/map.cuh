#pragma once


#include <cuda_runtime.h>
#include "../dtypes/concept.h"
#include "../shapes/shape_concept.h"
#include "../devices/includes.h"
#include "../references/includes.h"
#include "../cuda_helpers/kernel_launch.h"
#include "../threads/thread_context.h"
#include "../asserts/false_assert.h"
namespace NN {

	namespace Kernels {

		namespace Map {

			namespace Definitions {

				template<typename MapFunction, Shapes::IsShape Shape, typename T>
				__global__ void mapKernel(T *const arr) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < Shape::volume) {
						MapFunction::apply(arr[id]);
					}
				}
			}

			template<typename MapFunction, Shapes::IsShape Shape, typename T, Devices::Device Device>
			inline void map(References::MutableShapedDeviceReference<T, Shape, Device> arr, Threads::ThreadContext& ctx) {
				if constexpr (Devices::CUDADevice<Device>) {
					//CUDA path
					Definitions::mapKernel<MapFunction, Shape, T> << <CudaHelpers::computeGridSize(Shape::volume), CudaHelpers::computeBlockSize(Shape::volume), 0, ctx.stream >> > (arr);
				}
				else if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int id = 0; id < Shape::volume; id++) {
						MapFunction::apply(arr[id]);
					}
				}
				else {
					static_assert(struct_assert < Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}

		}
	}
}