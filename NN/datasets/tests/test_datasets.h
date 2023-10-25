#pragma once
#include "../interface/dataset.h"
#include "../../shapes/includes.h"

namespace NN {
	namespace Datasets {
		namespace Tests {

			template<typename T, Shapes::IsShape Shape = Shapes::unit<1>>
			class IdentityDataset : public Interfaces::Dataset<T, T, Shape, Shape> {
			private:
				virtual void getTest(unsigned long runId, T* input, T* output) override {
					for (int i = 0; i < Shape::volume; i++) {
						T val = (T)(rand() % 100) / 10;
						input[i] = val;
						output[i] = val;
					};
				}
				virtual void getTrainingSample(unsigned long runId, T* input, T* output) override {
					return getTest(runId, input, output);
				}
			public:
				IdentityDataset() : Interfaces::Dataset<T, T, Shape, Shape>() {}
			};

			template<typename T, Shapes::IsShape Shape = Shapes::unit<1>>
			class SquaredDataset : public Interfaces::Dataset<T, T,Shape, Shape> {
			private:
				virtual void getTest(unsigned long runId, T* input, T* output) override {
					for (int i = 0; i < Shape::volume; i++) {
						T val = (T)(rand() % 100) / 10;
						input[i] = val;
						output[i] = (T)pow(val, 2);
					};
				}
				virtual void getTrainingSample(unsigned long runId, T* input, T* output) override {
					return getTest(runId, input, output);
				}
			public:
				SquaredDataset() : Interfaces::Dataset<T, T,Shape,Shape>() {}
			};








			//a specific test to check backprop values
			template<typename T>
			class BackpropTest : public Interfaces::Dataset<T, T, Shapes::Shape<2>, Shapes::Shape<2>> {
			private:
				virtual void getTest(unsigned long runId, T* input, T* output) override {
					std::cerr << "There are no tests for BackpropTest database\n";
					throw;
				}
				virtual void getTrainingSample(unsigned long runId, T* input, T* output) override {
					input[0] = 0.05;
					input[1] = 0.1;
					output[0] = 0.01;
					output[1] = 0.99;
				}
			public:
				BackpropTest() : Interfaces::Dataset<T, T, Shapes::Shape<2>, Shapes::Shape<2>>() {}

			};


		}
	}
}