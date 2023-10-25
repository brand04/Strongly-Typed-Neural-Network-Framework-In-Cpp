#pragma once
#include "forward_declaration.h"
#include "../../references/includes.h"
#include "../../dtypes/concept.h"
#include <cuda_runtime.h>
#include <string>
namespace NN {
	namespace Storage {
		namespace Unsafe {
			//cpu specialization

		
			template<typename T, size_t size>
			class UnsafeStore<T, size, Devices::CPU> {
			private:
				T* ptr;



				template<size_t strLen>
				void throwIfError(cudaError_t error, const char(&identifier)[strLen]) const {
					if (error != cudaSuccess) {
						std::cerr << "Error during " << identifier << " - " << cudaGetErrorName(error) << " : " << cudaGetErrorString(error) << "\n";
						throw error;
					}
				}
			public:
				UnsafeStore() {
					ptr = reinterpret_cast<T*>(malloc(sizeof(T) * size));
					//ptr = reinterpret_cast<T*>(_aligned_malloc(sizeof(T) * size, alignof(T)));
					if (ptr == nullptr) {
						std::cerr << "Couldn't allocate enough CPU memory\n";
						throw;
					}
				}

				const T* asImmutable() const {
					return ptr;
				}

				T* asMutable() {
					return ptr;
				}

				//TODO: switch to thread contexts as input rather than a cuda stream


				//copy a slice of the store to a cuda store
				template<Devices::CUDADevice target>
				inline void copyTo(UnsafeStore<T, size, target>& other, size_t offset, size_t length, cudaStream_t stream) const {
					cudaError_t error;

					error = cudaSetDevice(target::deviceId);
					throwIfError(error, "cuda set device");

					error = cudaMemcpyAsync((other.asMutable()) + offset, ptr + offset, length, cudaMemcpyHostToDevice, stream);
					throwIfError(error, "cuda Memcpy");
				}

				//copy a slice of the store to another cpu store
				template<Devices::CPUDevice target>
				inline void copyTo(UnsafeStore<T, size, target>& other, size_t offset, size_t length) const {
					memcpy(other.asMutable() + offset, ptr + offset, length);
				}

				//copy the entire store to a gpu device's store
				template<Devices::CUDADevice target>
				inline void copyTo(UnsafeStore<T, size, target>& other, cudaStream_t stream)  const {
					copyTo(other, 0, size, stream); //copy everything
				}

				//copy the entire store to a gpu device's store
				template<Devices::CPUDevice target>
				inline void copyTo(UnsafeStore<T, size, target>& other)  const {
					copyTo(other, 0, size); //copy everything
				}



			

				~UnsafeStore()
				{
					free(ptr);
					//_aligned_free(ptr);
				}

			};

		}
	}
}