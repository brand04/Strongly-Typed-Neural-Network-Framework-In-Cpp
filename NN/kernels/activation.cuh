#pragma once
#include "../dtypes/concept.h"
#include "../shapes/shape_concept.h"
#include "../devices/includes.h"
#include "../references/includes.h"
#include "../cuda_helpers/kernel_launch.h"
#include "../threads/thread_context.h"
#include "../asserts/false_assert.h"
#include <cuda_runtime.h>
namespace NN {
	namespace Kernels {
		namespace ActivationFunction {

			namespace Definitions {

				template<typename ActivationFunction, Shapes::IsShape Shape , typename T>
				__global__  void ActivationFunctionKernel(T* const outputs) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < Shape::volume) {
						outputs[id] = ActivationFunction::apply(outputs[id]);
					}
				}
				
				template<typename ActivationFunction, Shapes::IsShape Shape , typename T>
				__global__  void ActivationFunctionBackpropKernel(T* const postNodeDeltas, T const *const outputs) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < Shape::volume) {
						postNodeDeltas[id] *= ActivationFunction::derivative(outputs[id]);
					}
				}
			}

			template<typename ActivationFunction, typename T, Shapes::IsShape Shape, Devices::Device Device>
			inline void forward(References::MutableShapedDeviceReference<T, Shape, Device> outputs, Threads::ThreadContext& ctx) {
				if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int id = 0; id < Shape::volume; id++) {
						outputs[id] = ActivationFunction::apply(outputs[id]);
					}
				}
				else if constexpr (Devices::CUDADevice<Device>) {
					Definitions::ActivationFunctionKernel<ActivationFunction, Shape, T> << <CudaHelpers::computeGridSize(Shape::volume), CudaHelpers::computeBlockSize(Shape::volume), 0, ctx.stream >> > (outputs);
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}

			template<typename ActivationFunction, typename T, Shapes::IsShape Shape, Devices::Device Device>
			inline void backward(References::MutableShapedDeviceReference<T, Shape, Device> postNodeDeltas, References::ImmutableShapedDeviceReference<T,Shape,Device> outputs, Threads::ThreadContext& ctx) {
				if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int id = 0; id < Shape::volume; id++) {
						postNodeDeltas[id] *= ActivationFunction::derivative(outputs[id]);
					}
				}
				else if constexpr (Devices::CUDADevice<Device>) {
					Definitions::ActivationFunctionBackpropKernel<ActivationFunction, Shape, T> << <CudaHelpers::computeGridSize(Shape::volume), CudaHelpers::computeBlockSize(Shape::volume), 0, ctx.stream >> > (postNodeDeltas,outputs);
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}


		}
	}
}