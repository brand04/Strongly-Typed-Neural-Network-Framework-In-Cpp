#pragma once
#include "../cuda_helpers/kernel_launch.h"
#include "../cuda_helpers/rand.h"
#include "../cuda_helpers/rand_kernels.cuh"
#include "../dtypes/concept.h"
#include "../references/includes.h"
#include "../devices/includes.h"
#include "../threads/thread_context.h"
#include "../asserts/false_assert.h"
namespace NN {
	namespace Kernels {
		namespace Initialization {


			template<Dtypes::Dtype Dtype, Shapes::IsShape WeightShape, Shapes::IsShape BiasShape, Devices::Device Device>
			void intialize(
				References::MutableShapedDeviceReference<typename Dtype::Type, WeightShape, Device> weights,
				References::MutableShapedDeviceReference<typename Dtype::Type, BiasShape, Device> biases,
				Threads::ThreadContext& ctx
			) {
				if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int i = 0; i < WeightShape::volume; i++) {
						weights[i] = Dtype::init(rand());
					}
					for (unsigned int i = 0; i < BiasShape::volume; i++) {
						biases[i] = Dtype::init(rand());
					}
				}
				else if constexpr (Devices::CUDADevice<Device>) {
					const static constexpr size_t numStates = std::min(CudaHelpers::MAX_BLOCK_SIZE * CudaHelpers::DESIRED_GRID_SIZE, (size_t)(WeightShape::volume + BiasShape::volume));
					curandStateXORWOW_t* states = CudaHelpers::initCurand<Device, std::min(CudaHelpers::MAX_BLOCK_SIZE * CudaHelpers::DESIRED_GRID_SIZE, (size_t)(WeightShape::volume + BiasShape::volume))>(ctx);
					CudaHelpers::initializeWeights<Dtype, numStates, WeightShape, BiasShape> << <CudaHelpers::computeGridSize(numStates), CudaHelpers::computeBlockSize(numStates), 0, ctx.stream >> > (weights, states, biases);
					CudaHelpers::freeCurand(states, ctx);
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}
		}
	}
}