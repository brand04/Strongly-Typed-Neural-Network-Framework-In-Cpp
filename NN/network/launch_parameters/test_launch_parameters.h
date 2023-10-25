#pragma once
#include "../../storage/network_io/test_store.h"
#include "../../shapes/includes.h"
#include "../../helpers/array_to_string.h"
#include <string>
namespace NN {
	namespace LaunchParameters {

		/// <summary>
		/// Handles the testing of a network
		/// </summary>
		/// <typeparam name="OutputType">The underlying type of the output</typeparam>
		/// <typeparam name="InputType">The underlying type of the input (e.g. double, uint8_t, int)</typeparam>
		/// <typeparam name="InputShape">The shape of the input</typeparam>
		/// <typeparam name="OutputShape">The shape of the output</typeparam>
		template<typename InputType, typename OutputType, Shapes::IsShape InputShape, Shapes::IsShape OutputShape>
		struct TestLaunchParameters {
			//place to hold the results
			Storage::NetworkIO::TestStore<InputType, OutputType, InputShape, OutputShape> results;
			//number of testing batches
			unsigned int batches = 1;
			//size of the batches, larger -> threads can run independently for longer
			unsigned int batchSize = 1;

			TestLaunchParameters(const unsigned int batches, const unsigned int batchSize) : results(batches* batchSize) {
				this->batches = batches;
				this->batchSize = batchSize;
			}


			/// <summary>
			/// returns a string containing a selection of samples from the test, inluding the input, computed outputs, and expected outputs
			/// </summary>
			/// <param name="numberOfSamples">The number of samples to return</param>
			/// <param name="random">whether the samples selected should be random or not</param>
			/// <returns>std::string containing formatted results</returns>
			std::string displayResults(unsigned int numberOfSamples, bool random = false) {
				std::stringstream s;
				for (unsigned int i = 0; i < numberOfSamples; i++) {
					unsigned int sampleId;
					if (!random) sampleId = i;
					else {
						sampleId = (rand() % (batches * batchSize));
					}
					
						s << "Sample " << std::to_string(sampleId) << " : input:\n "; 
						Helpers::arrayToString<InputType, InputShape>(s, "", results.inputs + (sampleId*InputShape::volume));
						s << "\noutput:\n ";
						Helpers::arrayToString<OutputType, OutputShape>(s, "", results.computed + (sampleId*OutputShape::volume));
						s << "\nexpected:\n";
						Helpers::arrayToString<OutputType, OutputShape>(s, "", results.expected + (sampleId * OutputShape::volume));
						s << "\n";

					
				}
				return s.str();
			}
		};
	}
}