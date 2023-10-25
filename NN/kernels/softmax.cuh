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
		namespace Softmax {

			static constexpr unsigned int PARALLEL_CUDA_WORKERS = 8;
			namespace Definitions {

				template<Dtypes::Dtype Dtype, size_t size, bool stable, bool fast>
				__global__ void softmaxKernel(typename Dtype::Type const * const inputs, typename Dtype::Type* const outputs) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					
					__shared__ typename Dtype::Type sums[PARALLEL_CUDA_WORKERS+1]; // store partial sums for each worker thread
						
					if (id < PARALLEL_CUDA_WORKERS) {
						sums[id] = Dtype::additiveIdentity;

						if constexpr (stable) { //if stable, subtract sum(arr[0],arr[1],...,arr[n-1]) before exponentiation
							sums[id] = inputs[(size / PARALLEL_CUDA_WORKERS) * id];
							for (unsigned int i = 1 + ((size / PARALLEL_CUDA_WORKERS) * id); i < (id == PARALLEL_CUDA_WORKERS-1 ? size : ((size / PARALLEL_CUDA_WORKERS) * (id + 1))); i++) {
								sums[id] = max(sums[id],inputs[i]);
							}
							__syncthreads();
							if (id == 0) {
								sums[PARALLEL_CUDA_WORKERS] = sums[0];
								for (unsigned int i = 1; i < PARALLEL_CUDA_WORKERS; i++) {
									sums[PARALLEL_CUDA_WORKERS] = max(sums[PARALLEL_CUDA_WORKERS],sums[i]);
								}
							}
							__syncthreads();
							sums[id] = Dtype::additiveIdentity;
						}

						
						for (unsigned int i = (size / PARALLEL_CUDA_WORKERS) * id; i < (id == PARALLEL_CUDA_WORKERS -1 ? size : ((size / PARALLEL_CUDA_WORKERS) * (id+1))); i++) {
							if constexpr (stable) (outputs[i] = inputs[i] - sums[PARALLEL_CUDA_WORKERS]); // see above
							if constexpr (std::same_as<float, typename Dtype::Type> && !fast) {
								if constexpr (stable) outputs[i] = expf(outputs[i]);
								else outputs[i] = expf(inputs[i]);	
							}
							else if constexpr (std::same_as<double, typename Dtype::Type> && !fast) {
								if constexpr (stable) outputs[i] = exp(outputs[i]);
								else outputs[i] = exp(inputs[i]);
							}
							else if constexpr (std::is_floating_point_v<typename Dtype::Type> && fast) {
								if constexpr (stable) outputs[i] = __expf(outputs[i]);
								else outputs[i] = __expf(inputs[i]);
							}
							else {
								static_assert(struct_assert<Asserts::AssertFalseWith<Dtype>>, "Expected a floating point type");
							}
							sums[id] += outputs[i];
							
						}

						__syncthreads();
						if (id == 0) {
							for (unsigned int i = 1; i < PARALLEL_CUDA_WORKERS; i++) {
								sums[0] += sums[i]; //sum the partial sums
							}
						}
						__syncthreads();
						for (unsigned int i = (size / PARALLEL_CUDA_WORKERS) * id; i < (id == PARALLEL_CUDA_WORKERS -1 ? size : ( (size / PARALLEL_CUDA_WORKERS) * (id + 1))); i++) {
							outputs[i] /= (sums[0]+(Dtype::multiplicativeIdentity * 0.0000001)); //divide by the sum
						}
					}
				}

				template<Dtypes::Dtype Dtype, size_t size>
				__host__ __device__ inline void softmaxDerivativeKernel(typename Dtype::Type *const preNodeDeltas, typename Dtype::Type const * const postNodeDeltas, typename Dtype::Type const * const outputs, const unsigned int id) {

					typename Dtype::Type sum = Dtype::additiveIdentity;
					for (unsigned int i = 0; i < size; i++) {
						if (id == i) {
							sum += (outputs[id] * (1 - outputs[id]))*postNodeDeltas[id];
						}
						else {
							sum -= (outputs[i] * outputs[id]) * postNodeDeltas[i];
						}
					}


					preNodeDeltas[id] = sum;


				}

				template<Dtypes::Dtype Dtype, size_t size>
				__global__ void softmaxDerivativeKernelLaunch(typename Dtype::Type* const preNodeDeltas, typename Dtype::Type const * const postNodeDeltas,  typename Dtype::Type const* const outputs) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < size) {
						softmaxDerivativeKernel<Dtype, size>(preNodeDeltas, postNodeDeltas, outputs, id);
					}
				}
			}

			template<Dtypes::Dtype Dtype,  Shapes::IsShape Shape, Devices::Device Device, bool stable = false, bool fast = false>
			inline void softmax(References::ImmutableShapedDeviceReference<typename Dtype::Type, Shape, Device> inputs, References::MutableShapedDeviceReference<typename Dtype::Type, Shape,Device> outputs, Threads::ThreadContext& ctx) {
				
				if constexpr (Devices::CUDADevice<Device>) {
					Definitions::softmaxKernel<Dtype, Shape::volume,stable, fast> << <1, PARALLEL_CUDA_WORKERS, 0, ctx.stream >> > (inputs,outputs);
				}
				else if constexpr (Devices::CPUDevice<Device>) {

					typename Dtype::Type mx = Dtype::additiveIdentity;

					if constexpr (stable) {
						mx = inputs[0];
						for (unsigned int i = 1; i < Shape::volume; i++) {
							mx = max(mx, inputs[i]);
						}

						for (unsigned int i = 0; i < Shape::volume; i++) {
							outputs[i] = inputs[i] - mx;
						}
					}







					typename Dtype::Type sum = Dtype::additiveIdentity;
					for (unsigned int i = 0; i < Shape::volume; i++) {
						if constexpr (std::same_as<float, typename Dtype::Type>) {
							if constexpr (stable) outputs[i] = expf(outputs[i]); //use stabalized value
							else outputs[i] = expf(inputs[i]);
						}
						else if constexpr (std::same_as<double, typename Dtype::Type>) {
							if constexpr (stable) outputs[i] = exp(outputs[i]); //use stabalized value
							else outputs[i] = exp(inputs[i]);
						}
						else {
							static_assert(struct_assert<Asserts::AssertFalseWith<Dtype>>, "Expected floating type");
						}
						sum += outputs[i];
					}
					for (unsigned int i = 0; i < Shape::volume; i++) {
						outputs[i] /= sum;
					}
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
				}
			}

			template<Dtypes::Dtype Dtype, Shapes::IsShape Shape, Devices::Device Device>
			inline void softmaxDerivative(References::MutableShapedDeviceReference<typename Dtype::Type, Shape, Device> preNodeDeltas, References::ImmutableShapedDeviceReference<typename Dtype::Type, Shape, Device> postNodeDeltas, References::ImmutableShapedDeviceReference<typename Dtype::Type, Shape, Device> outputs, Threads::ThreadContext& ctx) {

				if constexpr (Devices::CUDADevice<Device>) {
					Definitions::softmaxDerivativeKernelLaunch<Dtype, Shape::volume> << <CudaHelpers::computeGridSize(Shape::volume), CudaHelpers::computeBlockSize(Shape::volume), 0, ctx.stream >> > (preNodeDeltas, postNodeDeltas, outputs);
				}
				else if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int id = 0; id < Shape::volume; id++) {
						Definitions::softmaxDerivativeKernel<Dtype, Shape::volume>(preNodeDeltas, postNodeDeltas, outputs,id);
					}
				}

			}
		}
	}
}