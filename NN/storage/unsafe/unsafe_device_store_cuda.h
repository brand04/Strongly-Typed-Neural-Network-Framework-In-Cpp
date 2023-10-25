#pragma once

#include "forward_declaration.h"
#include "../../references/includes.h"
#include <cuda_runtime.h>
#include <string>
namespace NN {
	namespace Storage {
		namespace Unsafe {

			//GPU CUDA specialization


			template<typename T, size_t size, Devices::CUDADevice device>
			class UnsafeStore<T, size, device> {
			private:
				T* ptr;

				
				//helper function for error outputs
				template<size_t strLen>
				void throwIfError(const cudaError_t error, const char (&identifier)[strLen]) const {
					if (error != cudaSuccess) {
						std::cerr << "Error during " << identifier << " - " << cudaGetErrorName(error) << " : " << cudaGetErrorString(error) << "\n";
						throw error;
					}
				}
			public:
				UnsafeStore() {
					cudaError_t error = cudaMalloc(&ptr, sizeof(T) * size);
					throwIfError(error, "GPU memory allocation");

				}

				const T* asImmutable() const {
					return ptr;
				}

				T* asMutable() {
					return ptr;
				}


				//copy a slice of the store to another store
				template<Devices::CPUDevice target>
				inline void copyTo(UnsafeStore<T, size, target>& other, size_t offset, size_t length, cudaStream_t stream) const {
					cudaError_t error;

					error = cudaSetDevice(device::deviceId);
					throwIfError(error, "cuda set device");

					error = cudaMemcpyAsync((other.asMutable()) + offset, ptr + offset, length, cudaMemcpyDeviceToHost, stream);
					throwIfError(error, "cuda Memcpy");
				}

				//TODO: peer-to-peer

				//copy the entire store to a gpu device's store
				template<Devices::Device target>
				inline void copyTo(UnsafeStore<T, size, target>& other, cudaStream_t stream) const {
					copyTo<target>(other, 0, size, stream); //copy everything
				}

				~UnsafeStore()
				{
					cudaError_t error = cudaFree(ptr);
					throwIfError(error, "cuda freeing");
				}


			};

		}
	}
}