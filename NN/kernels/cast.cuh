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

		namespace Cast {

			namespace Definitions {

				template<Shapes::IsShape Shape, typename From, typename To>
				__global__ void castKernel(From const* const from, To* const to) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < Shape::volume) {
						to[id] = static_cast<To>(from[id]);
					}
				}
			}

			template<Shapes::IsShape Shape, typename From, typename To, Devices::Device Device>
			inline void cast(References::ImmutableShapedDeviceReference<From, Shape, Device> from, References::MutableShapedDeviceReference<To, Shape, Device> to, Threads::ThreadContext& ctx) {
				if constexpr (Devices::CUDADevice<Device>) {
					//CUDA path
					Definitions::castKernel<Shape, From, To> << <CudaHelpers::computeGridSize(Shape::volume), CudaHelpers::computeBlockSize(Shape::volume), 0, ctx.stream >> > (from, to);
				}
				else if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int id = 0; id < Shape::volume; id++) {
						to[id] = static_cast<To>(from[id]);
					}
				}
				else {
					static_assert(struct_assert < Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}

		}
	}
}