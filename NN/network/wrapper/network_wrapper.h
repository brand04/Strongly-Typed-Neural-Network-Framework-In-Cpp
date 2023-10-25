#pragma once
#include <cuda_runtime.h>
#include <thread>
#include <fstream>
#include "../../concepts/layer_concept.h"
#include "../../storage/training_output_store.h"
#include "../../storage/training_store.h"

#include "../../threads/thread_context.h"
#include "../../tensors/tensors.h"
#include "../../shapes/includes.h"

#include "../../functions/error_measures/halved_squared_error.h"
#include "../launch_parameters/includes.h"
#include "../../datasets/interface/application.h"
#include "../../datasets/interface/dataset.h"


#include "../traits/trait_get.h"


#include <functional>
#include <iostream>



namespace NN {
	namespace Networks {
		/// <summary>
		/// Wraps a layer as a network, providing network functions such as training, testing and running
		/// </summary>
		/// <typeparam name="Layer">The Layer to wrap</typeparam>
		template<Layers::Layer Layer, typename Measure = ErrorMeasures::HalvedSquaredError>
		class NetworkWrapper {
		private:
			//some type aliases
			using InputShape = typename Traits::getLayerTraits<Layer>::InputShape;
			using OutputShape = typename Traits::getLayerTraits<Layer>::OutputShape;

			//device aliases
			using InputDevice = typename Traits::getLayerTraits<Layer>::InputDevice;
			using OutputDevice = typename Traits::getLayerTraits<Layer>::OutputDevice;

			static_assert(struct_assert<Asserts::AssertSameDevices<InputDevice, Devices::CPU>>, "The Input Device for a layer wrapped as a network must be CPU");
			static_assert(std::same_as<OutputDevice, Devices::CPU>, "The Output Device for a layer wrapped as a network must be CPU");
			
			//dtypes - wrappers around a type defining various functions
			using InputDtype = typename Traits::getLayerTraits<Layer>::InputDtype;
			using OutputDtype = typename Traits::getLayerTraits<Layer>::OutputDtype;
			//the underlying types
			using InputType = typename Traits::getLayerTraits<Layer>::InputType;
			using OutputType = typename Traits::getLayerTraits<Layer>::OutputType;

			static constexpr unsigned int threads = Traits::getLayerTraits<Layer>::nThreads;


			
	
			

			//the wrapped layer
			Layer wrapped;
			//stores

			Storage::TrainingStore<InputDtype, InputShape, threads, Devices::CPU> inputStore;
			Storage::TrainingOutputStore<OutputDtype, OutputShape, threads, Devices::CPU> outputStore;
			//error measure
			//statistics
			OutputType batchAverages[threads];
			//thread handlers
			std::thread threadObjects[threads];
			//thread exceptions
			cudaError_t threadExceptions[threads];
			//thread contexts
			Threads::ThreadContext threadContexts[threads];

		public:

			//provide the display interfaces for compatibility with layers
			static constexpr const StringLiteral name = StringLiteral("Network");

			template<typename NetworkT, typename prepender = StringT<'\t'>>
			//Concat the string types into a fixed compile-time string containing the network architecture
			static constexpr const FixedString fullname = StringCollection(NetworkT::name, " (", InputShape::string, " -> " , OutputShape::string, ") {\n ", Layer::template fullname<Layer,StringT<>::append<prepender>, prepender>, "\n}\n").fix(); //use a fixed string here rather than combining into a StringCollection like in Layers because this is the full network

			NetworkWrapper() :
				inputStore(),
				outputStore(),
				
				wrapped(inputStore.makeTensor(), outputStore.makeTensor()),
				threadContexts()
			{

				
				static_assert(threads!=0, "Cannot create a zero threads network");
				
				for (int i = 0; i < threads; i++) {
					batchAverages[i] = 0;

					threadContexts[i].threadId = i;
				}

			}

			void run(LaunchParameters::LaunchParameters<OutputType, OutputShape>& params, Datasets::Interfaces::Application<InputType,InputShape>* dataset) {
				for (unsigned int batch = 0; batch < params.batches; batch++) {

					//initialize threads
					for (unsigned int thread = 0; thread < threads; thread++) {
						threadExceptions[thread] = cudaSuccess;
						threadObjects[thread] = std::thread([this, thread, &params, batch, dataset]() -> void {
							try {
								srand(time(NULL)); //random-ness within thread
								int threadBatchSize = params.batchSize / threads;
								if (thread < params.batchSize % threads) threadBatchSize += 1; //account for remaining passes that are not equally splittable between threads
								for (unsigned int pass = 0; pass < threadBatchSize; pass++) {
									unsigned long runId = batch * params.batchSize + (pass * threads) + thread;
									dataset->get(runId, inputStore.data.asMutable(this->threadContexts[thread]));
									Layer::template forward<Layer>(wrapped,this->threadContexts[thread]);
									this->threadContexts[thread].synchronize(); //wait for any outstanding operations
									//request the computed outputs
									References::MutableShapedDeviceReference<OutputType, OutputShape, Devices::CPU> outputs = outputStore.data.asMutable(this->threadContexts[thread]);
									memcpy(params.outputs + (runId * OutputShape::volume), outputs, sizeof(OutputType) * OutputShape::volume);

								}
							}
							catch (cudaError_t e) {
								std::cerr << ("A cuda exception was encountered within a thread: " + std::string(cudaGetErrorString(e)) + "\n");
								threadExceptions[thread] = e;
							}
							catch (...) {
								std::cerr << "An unknown exception occured within a thread\n";
							}
						});
					}


					//wait for threads to be completed threads (synchronize main thread with the worker threads)
					for (int thread = 0; thread < threads; thread++) {
						threadObjects[thread].join();
						if (threadExceptions[thread] != cudaSuccess) throw threadExceptions[thread]; //TODO: display error
					}


				}
			}

			void test(LaunchParameters::TestLaunchParameters<InputType, OutputType, InputShape, OutputShape>& params, Datasets::Interfaces::Dataset<InputType, OutputType, InputShape, OutputShape>* dataset) { //TODO: std::vector?
				OutputType averageLoss = 0;
				for (unsigned int batch = 0; batch < params.batches; batch++) {
					//initialize threads
					for (unsigned int thread = 0; thread < threads; thread++) {
						threadExceptions[thread] = cudaSuccess;
						batchAverages[thread] = 0;
						threadObjects[thread] = std::thread([this, thread, batch, dataset, &params]() -> void {
							try {
								srand(time(NULL)); //random-ness within thread
								int threadBatchSize = params.batchSize / threads;
								if (thread < params.batchSize % threads) threadBatchSize += 1; //account for remaining passes that are not equally splittable between threads
								for (unsigned int pass = 0; pass < threadBatchSize; pass++) {
									unsigned long runId = (batch * params.batchSize) + (pass * threads) + thread;
									dataset->test(runId, inputStore.data.asMutable(threadContexts[thread]), outputStore.expected.asMutable(threadContexts[thread]));
									Layer::template forward<Layer>(wrapped,this->threadContexts[thread]);
									//request the computed outputs

									//wait for request to succeed
									this->threadContexts[thread].synchronize();
									//reset the output store
									References::ImmutableShapedDeviceReference<OutputType, OutputShape, Devices::CPU> outputs = outputStore.data.asImmutable(threadContexts[thread]);
									References::ImmutableShapedDeviceReference<OutputType, OutputShape, Devices::CPU> expected = outputStore.expected.asImmutable(threadContexts[thread]);
									References::ImmutableShapedDeviceReference<InputType, InputShape, Devices::CPU> inputs = inputStore.data.asImmutable(threadContexts[thread]);

									//copy inputs, computed outputs and expected outputs to the Test Store
									memcpy(params.results.inputs + (InputShape::volume * runId), inputs, sizeof(InputType) * InputShape::volume);
									memcpy(params.results.computed + (OutputShape::volume * runId), outputs, sizeof(OutputType) * OutputShape::volume);
									memcpy(params.results.expected + (OutputShape::volume * runId), expected, sizeof(OutputType) * OutputShape::volume);

									batchAverages[thread] += Measure::template average<OutputType>(outputs, expected, OutputShape::volume) / (threadBatchSize);
								}
							}
							catch (cudaError_t e) {
								std::cerr << ("A cuda exception was encountered within a thread: " + std::string(cudaGetErrorString(e)) + "\n");
								threadExceptions[thread] = e;
							}
							catch (...) {
								std::cerr << "An unknown exception occured within a thread\n";
							}
						});

					}


					//wait for threads to be completed threads (synchronize main thread with the worker threads)
					OutputType batchAverageLoss = 0;
					for (int thread = 0; thread < threads; thread++) {
						threadObjects[thread].join();
						if (threadExceptions[thread] != cudaSuccess) throw threadExceptions[thread]; //TODO: display error
						batchAverageLoss += batchAverages[thread] / threads;
					}

					averageLoss = ((averageLoss * (batch + 1)) + batchAverageLoss) / (batch + 2); //add the batch average to the running overall average

					params.results.averageLoss = averageLoss;
					std::cout << "Test Complete: Average Loss: " + std::to_string(averageLoss) + "\n";




				}
			}


			void train(LaunchParameters::TrainingLaunchParameters<OutputType>& params, Datasets::Interfaces::Dataset<InputType, OutputType, InputShape, OutputShape>* dataset) {
				unsigned int lastBatchReset = 0;

				OutputType averageLoss = 0;
				OutputType averageLossDecaying = 0;

				for (unsigned int batch = 0; batch < params.batches; batch++) {

					//initialize threads
					for (unsigned int thread = 0; thread < threads; thread++) {
						threadExceptions[thread] = cudaSuccess;
						batchAverages[thread] = 0;
						threadObjects[thread] = std::thread([this, thread, &params, batch, lastBatchReset, dataset]() -> void {
							try {
								srand(time(NULL)); //random-ness within thread

								

								int threadBatchSize = params.batchSize / threads;
								if (thread < params.batchSize % threads) threadBatchSize += 1; //account for remaining passes that are not equally splittable between threads
								for (unsigned int pass = 0; pass < threadBatchSize; pass+=params.bundleSize) {
									
									unsigned long runId = (batch * params.batchSize) + (pass * threads) + thread; //the run number within all threads


									dataset->train(runId, inputStore.data.asMutable(threadContexts[thread]), outputStore.expected.asMutable(threadContexts[thread]));
									Layer::template forward<Layer>(wrapped,threadContexts[thread]);
									this->threadContexts[thread].synchronize();

									References::ImmutableShapedDeviceReference<OutputType,OutputShape,Devices::CPU> outputs = outputStore.data.asImmutable(threadContexts[thread]);
									References::ImmutableShapedDeviceReference<OutputType, OutputShape, Devices::CPU> expected = outputStore.expected.asImmutable(threadContexts[thread]);
									References::MutableShapedDeviceReference<OutputType, OutputShape, Devices::CPU> deltas = outputStore.deltas.asMutable(threadContexts[thread]);

									//compute errors
									for (int i = 0; i < OutputShape::volume; i++) {
										deltas[i] = Measure::derivative(outputs[i], expected[i]);
									}
									
									Layer::template backward<Layer>(wrapped, threadContexts[thread]);
									OutputType passLoss = Measure::template average<OutputType>(outputs, expected, OutputShape::volume); //the averaged loss for this pass
									if constexpr (std::is_floating_point_v<OutputType>){
										if (isnan(passLoss)) { // throw early so NaN does not get trained on
											std::cout << "NaN encountered - training divergence - stopping training\n";
											throw;
										}
									}
									if (passLoss > 20) {
										int i = 1;
									}
									else {
										int i = 1;
									}
									batchAverages[thread] += passLoss / threadBatchSize;


									//wait for backwards pass to succeed
									threadContexts[thread].synchronize();

									Layer::template modifyWeights<Layer>(wrapped, threadContexts[thread]);

								}
							}
							catch (cudaError_t e) {
								std::cerr << ("A cuda exception was encountered within a thread: " + std::string(cudaGetErrorString(e)) + "\n");
								threadExceptions[thread] = e;
							}
							catch (...) {
								std::cerr << "An unknown exception occured within a thread\n";
							}
							});
					}


					//wait for threads to be completed threads (synchronize main thread with the worker threads)
					OutputType batchAverage = 0;
					for (int thread = 0; thread < threads; thread++) {
						threadObjects[thread].join();
						if (threadExceptions[thread] != cudaSuccess) throw threadExceptions[thread]; //TODO: display error
						batchAverage += batchAverages[thread] / threads;
					}

					averageLoss = ((averageLoss * (batch % params.averageLifetimeBatches)) + batchAverage) / ((batch % params.averageLifetimeBatches) + 1);
					averageLossDecaying = ((averageLossDecaying + batchAverage) * params.averageLifetimeBatches) / (params.averageLifetimeBatches + 1);

					//Check if this batch requires outputting average loss report
					if (batch % params.batchesPerLossReport == 0) {


						if (params.includeDecayingAverage) {
							std::cout << "Batch " + std::to_string(batch) + ": loss =  " + std::to_string(averageLoss) + " (decaying loss: " + std::to_string(averageLossDecaying / ((batch + 1) < params.averageLifetimeBatches ? (batch+1) : (params.averageLifetimeBatches))) + ") \n";
						}
						else {
							std::cout << "Batch " + std::to_string(batch) + ": loss =  " + std::to_string(averageLoss) + "\n";
						}
					}

					if (params.batchesPerWeightOutput != 0 && (batch % params.batchesPerWeightOutput) == 0 && (batch!=0 || params.batchesPerWeightOutput==1)) { //check if we ever output the weights, and if so if this is a batch requiring it
						std::cout << displayWeights();
					}

					

					if (params.averageLifetimeBatches != 0 && (batch + 1) % params.averageLifetimeBatches == 0) { //check if we ever reset the average and if so if this is a batch requiring it
						std::cout << "---- Resetting average ----\n";
						lastBatchReset = batch;
						if (averageLoss < params.stopLossThreshold) { //check if the average loss over that lifetime is low enough to stop
							if (params.batchesPerSave != 0) { //only save at end if we were saving during training
								if (params.overwriteSave) {
									saveAsBytes(params.saveLocation);
								}
								else {
									saveAsBytes(params.saveLocation + "-end");
								}
								std::cout << "--- Created Save ---\n";
							}
							std::cout << "Loss threshold passed, ending training early\n";
							break; //end training early
						}
						if (params.stopLossMax != 0 && averageLoss > params.stopLossMax) {
							std::cout << "Loss insurance threshold surpassed - ending training early\n";
							break; //end training early so that saves are not overwritten or similar
						}
						averageLoss = 0;
					}
					else if (params.lossResetThreshold != OutputDtype::multiplicativeIdentity && (params.lossResetThreshold + batchAverage < averageLoss || params.lossResetThreshold + averageLoss < batchAverage)) {
						std::cout << " ----- Large Loss difference detected - resetting average -----\n";
						lastBatchReset = batch;
						averageLoss = 0;
					}

					if (params.batchesPerSave != 0 && (batch % params.batchesPerSave) == 0 && (batch != 0)) {
						std::cout << " ----- Creating Save ----\n ";
						if (params.overwriteSave == true) {
							saveAsBytes(params.saveLocation);
						}
						else {
							saveAsBytes(params.saveLocation + "-batch" + std::to_string(batch));
						}
					}
				}
			}




			/* fill the weights with the contents of the handle*/
			inline void setWeights(void* handle) {
				uint8_t* handleAsBytes = reinterpret_cast<uint8_t*>(handle);
				wrapped.setWeights(handleAsBytes, threadContexts[0]); //cast to uint8_t for pointer arithmetic purposes
			}

			inline void initializeWeights() {
				Threads::ThreadContext& singleThread = threadContexts[0];
				wrapped.initializeWeights(singleThread);
				singleThread.synchronize();
			}

			inline std::string displayWeights() {
				std::stringstream s;
				wrapped.template displayWeights<Layer>(s, "", threadContexts[0]);
				return s.str();
			}

			inline void saveAsBytes(std::string file) {
				std::ofstream f(file, std::ios::out | std::ios::binary);
			
				Layer::saveAsBytes(wrapped, f, threadContexts[0]);
				f.close();
			}

			inline void readFromBytes(std::string file) {
				std::ifstream f(file, std::ios::in | std::ios::binary);
				Layer::readFromBytes(wrapped, f, threadContexts[0]);
				f.close();
			}

			~NetworkWrapper()
			{
			}


		};
	}

}