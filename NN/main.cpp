
#include <cstdlib>
#include <stdio.h>
#include <sstream>
#include <iostream>

#include "datasets/mnist/mnist.h"

#include "./network/launch_parameters/includes.h"

#include "./layers/operations/display.h"

#include "./samples/mnist.h"








int main()
{
    //define some constants
    static constexpr bool PRINT_INCORRECTS = false;
    static constexpr unsigned int RESULTS_TO_DISPLAY = 50;
    static constexpr unsigned int TRAINING_BATCH_SIZE = 1000;
    static constexpr unsigned int TRAINING_BATCHES = 100000;

    const char MNIST_TEST_LABELS_PATH[] = "path/to/test/labels";
    const char MNIST_TEST_PIXELS_PATH[] = "path/to/test/images";
    const char MNIST_TRAINING_LABELS_PATH[] = "path/to/training/labels";
    const char MNIST_TRAINING_PIXLES_PATH[] = "path/to/training/images";


    srand(time(NULL));

    
    //Alias the type of our network (in this case using predefined types from ./smaples/mnist.h
    using Network = NN::MNIST_SMALL;

    //initialize the network
    Network network;
  
    //output the architecure to stdout
    std::string_view stringDisplay = NN::display<Network>;
    std::cout << NN::display<Network>;

    //if training, intialize the network according to the datatype provided to each layer
    network.initializeWeights();
    
    //or read from bytes
    //network.readFromBytes("MNIST-XL.nn-batch200");
    
    //load the datasets into main memory (or use some alternate database which reads it from file as-required)
    NN::Datasets::MNIST<double> trainingDataset(MNIST_TRAINING_LABELS_PATH, MNIST_TRAINING_PIXLES_PATH);
    NN::Datasets::MNIST<double> testDataset(MNIST_TEST_LABELS_PATH, MNIST_TEST_PIXELS_PATH);

    //create training parameters (batches to train for, passes per batches)
    typename Network::TrainingParameters trainingParams(1000000, 1250);
    trainingParams.setAverageLifetime(200).setSavePeriod(50, "./models/MNIST-S.nn", false).setStoppingThreshold(0.005).setLossReportPeriod(1).setInsuranceThreshold(0.4).setLossResetThreshold(1000.0).setWeightDisplayPeriod(0).enableDecayingAverage(true); //configure options

    const int testSize = 10000;
    NN::LaunchParameters::TestLaunchParameters<uint8_t, double, NN::Shapes::Shape<28, 28>, NN::Shapes::Shape<1, 10>> testParams(1, testSize);

    
    network.test(testParams, &testDataset);
    if constexpr (RESULTS_TO_DISPLAY > 0) {
        std::cout << testParams.displayResults(50);
    }
    
   
   
    std::cout << "Score : " + std::to_string(NN::Datasets::MNIST<double>::evaluateTest(testParams, PRINT_INCORRECTS)) + "\n";


    network.train(trainingParams, &trainingDataset);
    

    return 0;
}

