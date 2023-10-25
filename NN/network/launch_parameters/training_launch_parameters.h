#pragma once
#include "../../shapes/includes.h"

namespace NN {
	namespace LaunchParameters {
		/// <summary>
		/// Paramters determing the training process
		/// </summary>
		/// <typeparam name="T">The underlying type (e.g. double, uint8_t, int) of the output</typeparam>
		template <typename T>
		struct TrainingLaunchParameters {
			//number of training batches
			unsigned int batches = 1;
			//size of the batches, larger -> threads can run independently for longer
			unsigned int batchSize = 1;
			//how long before an average loss is displayed
			unsigned int batchesPerLossReport = 1;

			//how many batches before we commit a save to disk (checkpoint)
			unsigned int batchesPerSave = 1000;

			//whether to make unique saves for each time saving or to save only to a single file and overwrite as needed
			bool overwriteSave = false;

			//the location to save to
			std::string saveLocation = "";

			//after how many batches do we reset the average
			unsigned int averageLifetimeBatches = 5000;

			//if the loss differs by an amount > this value, reset the average
			T lossResetThreshold = 0;

			//Another average measure -> Will persist across average resets but is less stable than a summation average
			bool includeDecayingAverage = false;

			//If the average loss over batchLifetime is less than this value, halt training
			T stopLossThreshold = 0;

			//if the average loss over batchLifetime is greater than this value, halt training
			T stopLossMax = 0;

			//after how many batches should the weights be output
			unsigned int batchesPerWeightOutput = 0;

			//attempt to bundle together the errors of this many outputs before backpropogation
			unsigned int bundleSize = 1;

			TrainingLaunchParameters(unsigned int numberOfBatches, unsigned int sizeOfBatches)
			{
				batches = numberOfBatches;
				batchSize = sizeOfBatches;

			}

			TrainingLaunchParameters<T>& setLossReportPeriod(unsigned int period) {
				batchesPerLossReport = period;
				return *this;
			}

			TrainingLaunchParameters<T>& enableDecayingAverage(bool enable) {
				includeDecayingAverage = enable;
				return *this;
			}

			TrainingLaunchParameters<T>& setSavePeriod(unsigned int period, std::string saveLocation, bool overwriteSave = true) {
				batchesPerSave = period;
				this->saveLocation = saveLocation;
				this->overwriteSave = overwriteSave;
				return *this;
			}

			TrainingLaunchParameters<T>& setAverageLifetime(unsigned int lifetime) {
				averageLifetimeBatches = lifetime;
				return *this;
			}

			TrainingLaunchParameters<T>& setStoppingThreshold(T value) {
				stopLossThreshold = value;
				return *this;
			}

			//set a loss value as an insurance that training does not cause a significant reduction in the networks abilities
			TrainingLaunchParameters<T>& setInsuranceThreshold(T value) {
				stopLossMax = value;
				return *this;
			}

			TrainingLaunchParameters<T>& setWeightDisplayPeriod(unsigned int period) {
				batchesPerWeightOutput = period;
				return *this;
			}

			TrainingLaunchParameters<T>& setBundleSize(unsigned int bundleSize) {
				if (bundleSize == 0) {
					std::cerr << "Bundle size must not be 0" << std::endl;
					throw;
				}
				else {
					this->bundleSize = bundleSize;
				}
				return *this;
			}

			TrainingLaunchParameters<T>& setLossResetThreshold(T threshold) {
				this->lossResetThreshold = threshold;
				return *this;
			}
			
			
		};
	}
}