#pragma once
#include <iostream>

namespace NN {
	namespace LaunchParameters {

		/// <summary>
		/// Paramters for running the network only forwards, storing the outputs
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <typeparam name="OutputShape"></typeparam>
		template<typename T, typename OutputShape>
		struct LaunchParameters {
			unsigned int batches = 1;
			unsigned int batchSize = 1;
			T* outputs = nullptr;
			LaunchParameters(unsigned int numberOfBatches, unsigned int sizeOfBatches) {
				batches = numberOfBatches;
				batchSize = sizeOfBatches;
				outputs = (T*)malloc(sizeof(T) * OutputShape::volume * batches * batchSize);
				if (outputs == nullptr) {
					std::cerr << "An exception has occured during allocation of output storage\n";
					throw; //TODO: better exception
				}
			}

			~LaunchParameters()
			{
				if (outputs != nullptr) {
					free(outputs);
					outputs = nullptr;
				}
			}
		};


	}
}