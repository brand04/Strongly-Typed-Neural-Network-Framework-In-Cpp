#pragma once
#include <cuda_runtime.h>
#include "./copy.cuh"
#include "../cuda_helpers/kernel_launch.h"
namespace NN {
	namespace Kernels {
		namespace CopyExpansion {

			namespace Definitions {

				template<typename Dtype, typename ...Ts>
				__host__ __device__ inline void copyExpansionBackwardImpl( typename Dtype::Type* const preNodeDeltas, Ts const* const ... postNodeDeltas, const unsigned int id) {
					preNodeDeltas[id] = (Dtype::additiveIdentity + ... + postNodeDeltas[id]);
					
				}

				template<typename Shape, typename Dtype, typename ...Ts>
				__global__ void copyExpansionBackwardsLaunch(typename Dtype::Type* const preNodeDeltas, Ts const* const ... postNodeDeltas) {
					const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
					if (id < Shape::volume) {
						copyExpansionBackwardImpl<Dtype, Ts...>(preNodeDeltas, postNodeDeltas..., id);
					}
				}

			}


			//copy expansion is the duplication of a single shape to many places so that the effective shape after the operation is Shape::expand<n>


			/// <summary>
			/// Copies the contents of src into each of dsts
			/// </summary>
			/// <typeparam name="T"> The type (not dtype) </typeparam>
			/// <typeparam name="Device">The Device of the references</typeparam>
			/// <typeparam name="...Shapes">A pack of equal shapes (as well as equal to shape0)</typeparam>
			/// <typeparam name="Shape0"></typeparam>
			/// <param name="src">source location</param>
			/// <param name="...dsts">destination locations</param>
			template<typename  T, typename Device,typename Shape0, typename ... Shapes>
			inline void copyExpansionForward(References::ImmutableShapedDeviceReference<T, Shape0, Device> src, Threads::ThreadContext& ctx, References::MutableShapedDeviceReference<T,Shapes, Device>... dsts) {
				((Copy::copy<T, Shape0, Device>(static_cast<void const*const>(src.ptr), dsts, ctx)), ...);
			}

			/// <summary>
			/// backpropogates errors from an expansion
			/// </summary>
			template<typename Dtype, typename Device, typename Shape0, typename ... Shapes>
			inline void copyExpansionBackward(References::MutableShapedDeviceReference<typename Dtype::Type, Shape0, Device> preNodeDeltas, Threads::ThreadContext& ctx, References::ImmutableShapedDeviceReference<typename Dtype::Type, Shapes, Device>... postNodeDeltas) {
				if constexpr (Devices::CUDADevice<Device>) {
					Definitions::copyExpansionBackwardsLaunch<Shape0, Dtype, typename std::pair<typename Dtype::Type, Shapes>::first_type...> << <CudaHelpers::computeGridSize(Shape0::volume), CudaHelpers::computeBlockSize(Shape0::volume), 0, ctx.stream >> > (preNodeDeltas, postNodeDeltas...);
				}
				else if constexpr (Devices::CPUDevice<Device>) {
					for (unsigned int id = 0; id < Shape0::volume; id++) {
						Definitions::copyExpansionBackwardImpl<Dtype, typename std::pair<typename Dtype::Type, Shapes>::first_type...>(preNodeDeltas, postNodeDeltas..., id);
					}
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Device not supported - Only CPU and CUDA supported");
				}
			}

		}

		
	}
}