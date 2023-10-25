#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace Threads {

		/// <summary>
		/// Container for all the information a thread might need for asynchronous operations
		/// Should be passed by reference
		/// </summary>
		struct ThreadContext {
			unsigned int threadId; //a unique identifier continuous from 0
			cudaStream_t stream = nullptr; //if using cuda, should be initialized in prepareCuda
			unsigned long long syncEpoch; // a counter of how many times the stream has been synchronized - used in randomness calculations to prevent re-seeding due to operations completing too fast

			/* Array initialization, does not initalize a unique threadId*/
			ThreadContext() {
				
			
			}
			ThreadContext(const unsigned int threadIdentifier) : threadId(threadIdentifier) {
			}

			//Initialization using specific stream
			ThreadContext(const unsigned int threadIdentifier, cudaStream_t parentStream) : threadId(threadIdentifier), stream(parentStream) {}

			//waits for all processes in the stream to complete
			void synchronize() /* throw (cudaError_t) */ {
				if (stream != nullptr) {
					cudaError_t err = cudaStreamSynchronize(stream);
					if (err != cudaSuccess) {
						std::cerr << "Cuda Error during synchronization " << cudaGetErrorName(err) << " : " << cudaGetErrorString(err) << "\n";
						throw err;
					}
				}
				++syncEpoch;
			}

			//creates a stream if no stream already exists in this thread context
			void prepareCuda() {
				if (stream == nullptr) { //check not already intialized
					cudaError_t err = cudaStreamCreate(&stream);
					if (err != cudaSuccess) {
						std::cerr << "Error creating thread stream\n";
						throw err;
					}
				}
			}

			//destroys this thread, cleaning up any streams used by it
			~ThreadContext()
			{
				if (stream != nullptr) {
					cudaError_t err = cudaStreamDestroy(stream);
					if (err != cudaSuccess) {
						std::cerr << "Error destroying thread stream\n";
						throw err;
					}
				}
			}

		
		};
	}



}