#pragma once
#include "../interface/dataset.h"
#include "../../shapes/includes.h"
#include "../../helpers/endian.h"
#include "../../helpers/array_to_string.h"
#include "../../network/launch_parameters/includes.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <concepts>
#include <bit>
namespace NN {
	namespace Datasets {

		template<std::floating_point FloatT>
		class MNIST : public Interfaces::Dataset<uint8_t, FloatT, Shapes::Shape<28, 28>, Shapes::Shape<1,10>> {
		private:
			uint8_t* pixels;
			uint8_t* labels;
			unsigned int count;
			static constexpr bool DISPLAY_LABELS = false; //display the labels as we read them in
		public:
			virtual void getTrainingSample(unsigned long runId, uint8_t* input, FloatT* output) override {
				//unsigned int imageId = (runId % count);
				unsigned int imageId = rand() % count;
				memcpy(input, pixels + (imageId * Shapes::Shape<28, 28>::volume), sizeof(uint8_t) * Shapes::Shape<28, 28>::volume);
				for (unsigned int i = 0; i < Shapes::Shape<10>::volume; i++) {
					output[i] = (FloatT)0;
				}
				output[labels[imageId]] = (FloatT)1;
			}

			virtual void getTest(unsigned long runId, uint8_t* input, FloatT* output) override {
				unsigned int imageId = (runId % count); // dont randomize
				//unsigned int imageId = rand() % count;
				memcpy(input, pixels + (imageId * Shapes::Shape<28, 28>::volume), sizeof(uint8_t) * Shapes::Shape<28, 28>::volume);
				for (unsigned int i = 0; i < Shapes::Shape<10>::volume; i++) {
					output[i] = (FloatT)0;
				}
				output[labels[imageId]] = (FloatT)1;
			}


			MNIST(std::string labelPath, std::string imagePath) : Interfaces::Dataset<uint8_t, FloatT, Shapes::Shape<28, 28>, Shapes::Shape<1,10>>() {

				std::cout << "Attempting to load MNIST dataset from " << imagePath << " and "<< labelPath << "\n";


				std::ifstream imagesFile;
				std::ifstream labelsFile;

				try {
					imagesFile.open(imagePath, std::ios::in | std::ios::binary);
					labelsFile.open(labelPath, std::ios::in | std::ios::binary);
					if (imagesFile.fail()) {
						std::cerr << "Failed opening ImagesFile\n";
						throw;
					}
					if (labelsFile.fail()) {
						std::cerr << "Failed opening LabelsFile\n";
						throw;
					}
					unsigned int magic = 0;
					labelsFile.read(reinterpret_cast<char*>(&magic),sizeof(unsigned int));
					/*
					if (std::endian::native == std::endian::little) {
						magic = Helpers::flipEndian(magic);
						
					}
					*/
					magic = Helpers::fromEndian(magic, std::endian::big);
					if (magic != 2049) {
						std::cerr << "Warning : invalid label magic value : " << std::to_string(magic) << "\n";
					}
					labelsFile.read(reinterpret_cast<char*>(&count), sizeof(unsigned int));
					/*
					if (std::endian::native == std::endian::little) {
						count = Helpers::flipEndian(count);		
					}
					*/
					count = Helpers::fromEndian(count, std::endian::big);
					std::cout << "Labels file declares there are " << std::to_string(count) << " images\n";

					imagesFile.read(reinterpret_cast<char*>(&magic), sizeof(unsigned int));
					if (std::endian::native == std::endian::little) {
						magic = Helpers::flipEndian(magic);
					}
					if (magic != 2051) {
						std::cerr << "Warning : invalid image magic value : " << std::to_string(magic) <<  "\n";
					}
					int tmp = count;
					imagesFile.read(reinterpret_cast<char*>(&count), sizeof(unsigned int));
					/*
					if (std::endian::native == std::endian::little) {
						count = Helpers::flipEndian(count);
					}
					*/
					count = Helpers::fromEndian(count, std::endian::big);
					if (count != tmp) {
						std::cerr << "Warning : image declaration of the number of images does not match the label's - images file declared  " << std::to_string(count) << " images \n";
					}
					int x, y;
					imagesFile.read(reinterpret_cast<char*>(&x), sizeof(unsigned int));
					imagesFile.read(reinterpret_cast<char*>(&y), sizeof(unsigned int));
					/*
					if (std::endian::native == std::endian::little) {
						x = Helpers::flipEndian(x);
						y = Helpers::flipEndian(y);
					}
					*/
					x = Helpers::fromEndian(x, std::endian::big);
					y = Helpers::fromEndian(y, std::endian::big);
					if (x != 28 || y != 28) {
						std::cerr << "Warning : the dimensions should be 28 by 28, received: " << std::to_string(x) << " by " << std::to_string(y) << "\n";
					}

					std::cout << "Loading labels...\n";

					pixels = (uint8_t*)malloc(Shapes::Shape<28, 28>::volume * sizeof(uint8_t) * count);
					labels = (uint8_t*)malloc(sizeof(uint8_t) * count);

					if (pixels == nullptr || labels == nullptr) {
						std::cerr << "Error allocatiing memory for images\n";
						throw;
					}

					size_t offset = 0;
					for (unsigned int i = 0; i < count; i++) {
						labelsFile.read(reinterpret_cast<char*>(labels + offset),sizeof(uint8_t));
						*(labels + offset) = Helpers::fromEndian(*(labels + offset), std::endian::big);
						/*if (std::endian::native == std::endian::little) {
							*(labels+offset) = Helpers::flipEndian(*(labels+offset));
						}
						*/
						if constexpr (DISPLAY_LABELS) {
							std::cout << "Label " << std::to_string(offset) << " : " << std::to_string(*(labels + offset)) << "\n";
						}
						offset++;
					}
					std::cout << "Loading images...\n";
					offset = 0;
					for (unsigned int i = 0; i < count; i++) {
						imagesFile.read(reinterpret_cast<char*>(pixels + (offset*Shapes::Shape<28,28>::volume)),sizeof(uint8_t)*Shapes::Shape<28,28>::volume);
						/*
						if (std::endian::native == std::endian::little) {
							*(pixels + (offset*Shapes::Shape<28,28>::volume)) = Helpers::flipEndian(*(pixels + (offset*Shapes::Shape<28,28>::volume)));
						}
						*/
						*(pixels + (offset * Shapes::Shape<28, 28>::volume)) = Helpers::fromEndian(*(pixels + (offset * Shapes::Shape<28, 28>::volume)),std::endian::big);
						offset++;
					}
					std::cout << "Loaded dataset\n";



				}
				catch (const std::ifstream::failure& e) {
					std::cerr << "An error occured opening the file : "<<e.what() << "\n";
					
				}

			}

			/// <summary>
			/// Given results for a MNIST dataset, compute the percentage of the passes which successfully (assuming the highest logit taken) predicts the correct digit
			/// Can output incorrect computed/expected outputs to stdout for further inspection
			/// </summary>
			static const double evaluateTest(const NN::LaunchParameters::TestLaunchParameters<uint8_t, FloatT, NN::Shapes::Shape<28, 28>, NN::Shapes::Shape<1, 10>> &testParams, const bool outputIncorrects = false) {
				const unsigned int testSize = testParams.batches * testParams.batchSize;
				double score = 0.0;
				for (unsigned int i = 0; i < testSize; i++) {
					int selectedComputed = 0;
					double maxComputed = testParams.results.computed[i * 10];
					int selectedExpected = 0;
					double maxExpected = testParams.results.expected[i * 10];
					for (unsigned int j = 1; j < 10; j++) {
						if (testParams.results.computed[i * 10 + j] > maxComputed) {
							selectedComputed = j;
							maxComputed = testParams.results.computed[i * 10 + j];
						}
						if (testParams.results.expected[i * 10 + j] > maxExpected) {
							selectedExpected = j;
							maxExpected = testParams.results.expected[i * 10 + j];
						}
					}
					if (selectedComputed == selectedExpected) {
						score += (1.0 / testSize);
					}
					else {
						if (outputIncorrects) {
							std::cout << "Incorrect - selected: " << std::to_string(selectedComputed) << ", expected: " << std::to_string(selectedExpected) << "\n" << NN::Helpers::arrayToString<double, NN::Shapes::Shape<1, 10>>(&testParams.results.computed[i * 10]) << "\n" << NN::Helpers::arrayToString<double, NN::Shapes::Shape<1, 10>>(&testParams.results.expected[i * 10]) << "\n";
						}
					}
				}
				return score;
			}
			~MNIST()
			{
				if (pixels != nullptr) {
					free(pixels);
				}
				if (labels != nullptr) {
					free(labels);
				}
			}
		};
	}
}