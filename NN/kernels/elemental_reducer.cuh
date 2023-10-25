#pragma once
#include <cuda_runtime.h>
#include "../references/includes.h"
#include "../threads/thread_context.h"
#include "../cuda_helpers/kernel_launch.h"
#include "../cuda_helpers/set_device.h"
#include "../asserts/false_assert.h"
//equivalent to dot product if that joiner is used

namespace NN {
	namespace Kernels {
		namespace Reducers {
			namespace Definitions {
				template<typename Joiner, size_t size, typename ResultType, typename ... Types>
				__global__ void elementalReducerKernel(ResultType* const result, Types const * const... ptrs) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < size) {
						result[id] = Joiner::apply(ptrs[id]...);
					}
				}

				template<typename Joiner, size_t size, typename ResultType, typename ... Types>
				__global__ void elementalReducerBackpropKernel(Types* const... preNodeDeltas, ResultType const * const postNodeDeltas, ResultType const * const outputs, Types const* const... inputs) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < size) {
						((preNodeDeltas[id] = Joiner::derivative(outputs[id], inputs[id]) * postNodeDeltas[id]), ...);
					}
				}


			}

			template<typename Joiner, typename Dtype, typename Device, typename Shape0, typename ... Shapes>
			void elementalReduceForward(References::MutableShapedDeviceReference<typename Dtype::Type, Shape0, Device> output, Threads::ThreadContext& ctx, References::ImmutableShapedDeviceReference<typename Dtype::Type, typename Shapes, Device>... inputs) {
				if constexpr (Devices::CUDADevice<Device>) {
					CudaHelpers::setDevice<Device>();
					Definitions::elementalReducerKernel<Joiner, Shape0::volume, typename Dtype::Type, typename std::pair<typename Dtype::Type, Shapes>::first_type ...><<<CudaHelpers::computeGridSize(Shape0::volume), CudaHelpers::computeBlockSize(Shape0::volume), 0, ctx.stream>>>(output, inputs...);
				}
				else if constexpr (Devices::CPUDevice<Device>){
					
					for (unsigned int i = 0; i < Shape0::volume;i++) {
						output[i] = Joiner::apply(inputs[i]...);
					}
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}


			template<typename Joiner,  typename Dtype, typename Device, typename Shape0, typename ... Shapes>
			void elementalReduceBackward(
				References::ImmutableShapedDeviceReference<typename Dtype::Type, Shape0, Device> postNodeDeltas, 
				References::ImmutableShapedDeviceReference<typename Dtype::Type, Shape0, Device> outputs,
				Threads::ThreadContext& ctx,
				References::ImmutableShapedDeviceReference<typename Dtype::Type,  Shapes, Device>... inputs,
				References::MutableShapedDeviceReference<typename Dtype::Type,  Shapes, Device>... preNodeDeltas
			) {
				if constexpr (Devices::CUDADevice<Device>) {
					CudaHelpers::setDevice<Device>();
					//use std::pair to convert from a pack of shapes to a pack of types
					Definitions::elementalReducerBackpropKernel<Joiner, Shape0::volume, typename Dtype::Type, typename std::pair<typename Dtype::Type, Shapes>::first_type ...> << <CudaHelpers::computeGridSize(Shape0::volume), CudaHelpers::computeBlockSize(Shape0::volume), 0, ctx.stream >> > (preNodeDeltas..., postNodeDeltas, outputs, inputs...);
				}
				else if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int i = 0; i < Shape0::volume;i++) {
						((preNodeDeltas[i] = postNodeDeltas[i] * Joiner::derivative(outputs[i],inputs[i])) , ...); //fold over each preNodeDeltas arg and each input arg
					}
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}

		}
	}
}