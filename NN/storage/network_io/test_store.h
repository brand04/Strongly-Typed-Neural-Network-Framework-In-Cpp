#pragma once
#include "../../../shapes/includes.h"
#include <iostream>

namespace NN {
	namespace Storage {
		namespace NetworkIO {

			/// <summary>
			/// Stores the result of a test on a network
			/// </summary>
			/// <typeparam name="U">type of the input</typeparam>
			/// <typeparam name="V">type of the output</typeparam>
			/// <typeparam name="InputShape">Shape of the input</typeparam>
			/// <typeparam name="OutputShape">Shape of the output</typeparam>
			template<typename U, typename V, Shapes::IsShape InputShape, Shapes::IsShape OutputShape>
			struct TestStore {
				U* inputs;
				V* computed;
				V* expected;
				V averageLoss;

				TestStore(const unsigned int entries) {
					//TODO align
					inputs = (U*)malloc(InputShape::volume * entries * sizeof(U));
					computed = (V*)malloc(OutputShape::volume * entries * sizeof(V));
					expected = (V*)malloc(OutputShape::volume * entries * sizeof(V));
					if (computed == nullptr || inputs == nullptr || expected == nullptr) {
						std::cerr << "Failed to allocate storage for test\n";
						throw;
					}
				}

				~TestStore()
				{
					if (inputs != nullptr) free(inputs);
					if (computed != nullptr) free(computed);
					if (expected != nullptr) free(expected);
				}
			};
		}
	}
}