#pragma once

//Layer Wrappers (use same output/input memory)
#include "../layers/phantom_layers/activation_function.h" //wrapper applying activation functions
#include "../layers/phantom_layers/error_preprocessor.h" //manipulation of layer learning rates with respect to depth

//layers
#include "../layers/softmax/softmax.h" //softmax layer
#include "../layers/conv/biased_conv.h" //biased convoloution layer
#include "../layers/conv/conv.h" //unbiased convoloution layer
#include "../layers/overlay/biased_overlay.h" //biased overlay layer
#include "../layers/overlay/overlay.h" //unbiased overlay layer
#include "../layers/type_transformations/type_cast.h" //casting layer

//Functions

//Activation functions
#include "../functions/activation_functions/leaky_relu_activation.h" //leaky Relu activation
#include "../functions/activation_functions/logistic_activation.h" // logistic activation

//Unary functions
#include "../functions/map_functions/scale.h" //Used with ErrorPreprocessor

//Reducing functions (multivariate)
#include "../functions/reducers/dot_product.h" //combining during Concurrent Sequence
#include "../functions/reducers/summation.h" //combining during Concurrent Sequence



//Datatypes
#include "../dtypes/scaled_with_offset_integral_dtype.h" //weight initialization and definitions - need to initalize small to avoid NaN
#include "../dtypes/includes.h" //basic primitive->datatype wrappers

//Sequences
#include "../sequences/concurancy/concurrent_sequence.h" //Allow the same features to be processed in multiple ways and recombined
#include "../sequences/numbered_sequence.h" //layer sequencing (can just use a normal sequence rather than numbered)

//Error measures
#include "../functions/error_measures/cross_entropy_error.h" //doing a classification problem, so use the Cross Entropy Error

//Network object
#include "../network/network.h" //network functions


namespace NN {

    //contains MNIST network types

	namespace MNIST_SMALL_TYPES {

        //A simple layer to layer Deep neural network that casts the input, then runs 2 3x3 kernels over it (with positional biases), then a 7x7 kernel (again with positional biases), then a 5x5 kernel (ditto), before using this 12x12 shape as a kernel on a pre-determined set of weightings, then softmaxing the result
        //after each layer an activation function is applied (except the first and last)


        //can achieve 90% score, proving that BiasedConvoloution, BiasedOverlay, ActivationWrapper and Softmax are all capable of backpropogation successfully

        static constexpr bool USE_CUDA = true;
        static constexpr unsigned int threads = 1; //TODO: fix multithreading bug
        
        //define model weight type
        using WeightType = double;

        //define the dtype for training
        using WeightDatatype = NN::Dtypes::AsDtypeScaledWithOffset<WeightType, 0.02, 0.05>;
        using InputDatatype = NN::Dtypes::AsDtype<uint8_t>; //we receive a 28x28 grid of values between 0 and 255 - no weights on the cast layer so no need for control over initialization

        //define the device to process on (the network is auto-adjusted to start and end on CPU)
        using ProcessingDevice = std::conditional_t<USE_CUDA,NN::Devices::CUDA<0>, NN::Devices::CPU>; //determines the device based on the USE_CUDA constexpr

        using ConvAdjuster = NN::WeightModifiers::ClampedLinear<WeightType, 0.1, 0.0001>;
        using OverlayAdjuster = ConvAdjuster;
        //define the type of each individual layer
        using CastLayer = NN::Layers::Cast<InputDatatype, WeightDatatype, NN::Shapes::Shape<28, 28>, ProcessingDevice, threads>; //define a casting layer from the input type of 0-255 uints to the type of the weights (double)
        using Conv1 = NN::Layers::BiasedConvoloution<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<28, 28>, NN::Shapes::Shape<3, 3>, NN::Shapes::Shape<26, 26>, threads>;
        using Conv2 = NN::Layers::Phantom::ActivationWrapper<NN::Layers::BiasedConvoloution<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<26, 26>, NN::Shapes::Shape<3, 3>, NN::Shapes::Shape<24, 24>, threads, ConvAdjuster>, NN::Activations::LeakyReluActivation<>>; //wrap with LeakyRelu
        using Conv3 = NN::Layers::Phantom::ActivationWrapper<NN::Layers::BiasedConvoloution<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<24, 24>, NN::Shapes::Shape<3, 3>, NN::Shapes::Shape<22, 22>, threads, ConvAdjuster>, NN::Activations::LeakyReluActivation<>>;
        using Conv4 = NN::Layers::Phantom::ActivationWrapper<NN::Layers::BiasedConvoloution<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<22, 22>, NN::Shapes::Shape<7, 7>, NN::Shapes::Shape<16, 16>, threads, ConvAdjuster>, NN::Activations::LeakyReluActivation<>>;
        using Conv5 = NN::Layers::Phantom::ActivationWrapper<NN::Layers::BiasedConvoloution<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<16, 16>, NN::Shapes::Shape<5, 5>, NN::Shapes::Shape<12, 12>, threads, ConvAdjuster>, NN::Activations::LeakyReluActivation<>>;
        using CombinationLayer = NN::Layers::Phantom::ActivationWrapper<NN::Layers::BiasedOverlay<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<12, 12>, NN::Shapes::Shape<1, 10>, threads, OverlayAdjuster>, NN::Activations::LeakyReluActivation<>>;
        using SoftmaxLayer = NN::Layers::Softmax<WeightDatatype, NN::Shapes::Shape<1, 10>, ProcessingDevice, 1>; //run softmax on the outputs
        
        //all of the above now need to be sequenced, for this we can use NN::NumberedSequence - this causes CastLayer to be run first, then Conv1, then Conv2 -> Conv3 -> Conv4 -> Conv5 -> CombinationLayer -> SoftmaxLayer
        using Sequence = NN::NumberedSequence<CastLayer, Conv1, Conv2, Conv3, Conv4, Conv5, CombinationLayer, SoftmaxLayer>; //runs each layer in sequence from left to right, passing the outputs from the previous layer as the inputs of the next layer without data copying
        

        //now wrap the overall layer (in this case is just a single sequence) as a network and provide the error measure
        using Network = NN::Network<Sequence, NN::ErrorMeasures::CrossEntropyError>;
	}


    namespace MNIST_MEDIUM_TYPES {

        //A more complicated Network, testing the ability of Concurrent Sequence to successfully backpropogate and converge

        //Demonstrates how to sequence many things together, create concurrent sequences that themselves contain concurrent sequences, different ways of combining the concurrent sequences as well as the partial application of Layer templates using type aliases to reduce repetition
        //this could be taken further by using Traits::getLayerTraits<PrevLayer>::OutputShape to deduce the shape from the last layer you will apply

        //can achieve 94.1%, showing some improvement (though my choices of layers are more for general coverage in testing everything works rather than effectiveness)

        static constexpr bool USE_CUDA = true;
        static constexpr double DEPTH_VOLATILITY_FACTOR = 0.91; //by artificially deflating the errors of earlier layers we can provide a more stable search space for the later layers (because the earlier weights do not suddenly change by a relatively small amount that propogates to a large shift in values later)
        static constexpr unsigned int threads = 1;

        //Use weights of type double
        using WeightType = double;
        using WeightDatatype = NN::Dtypes::AsDtypeScaledWithOffset<WeightType, 0.001, 0.05>; //intialize them quite small with little variance

        using InputDatatype = NN::Dtypes::AsDtype<uint8_t>; //we receive a 28x28 grid of values between 0 and 255

        //define the device to process on (the network is auto-adjusted to start and end on CPU)
        using ProcessingDevice = std::conditional_t<USE_CUDA, NN::Devices::CUDA<0>, NN::Devices::CPU>; //determines the device based on the USE_CUDA constexpr

        //define a weight adjuster for Convoloution and Overlay, which clamps any changes and has a somewhat low learning rate (to adjust during training)
        using ConvAdjuster = NN::WeightModifiers::ClampedLinear<double, 0.01, 0.000001>;
        using OverlayAdjuster = NN::WeightModifiers::ClampedLinear<double, 0.01, 0.000001>;

        //Cast to WeightType
        using CastLayer = NN::Layers::Cast<InputDatatype, WeightDatatype, NN::Shapes::Shape<28, 28>, ProcessingDevice, 1>; //define a casting layer from the input type of 0-255 uints to the type of the weights (double)

        //define shorthand template alias
        template<typename InputShape, typename KernelShape>
        using BConv = NN::Layers::Phantom::ErrorPreprocessor<NN::Layers::BiasedConvoloution < WeightDatatype, ProcessingDevice, InputShape, KernelShape, typename InputShape::subtract<KernelShape>::add<NN::Shapes::unit<KernelShape::dimension>>, threads, ConvAdjuster>,NN::MapFunctions::Scale<WeightDatatype,DEPTH_VOLATILITY_FACTOR>>;

        template<typename InputShape, typename KernelShape>
        using Conv = NN::Layers::Phantom::ErrorPreprocessor<NN::Layers::BiasedConvoloution < WeightDatatype, ProcessingDevice, InputShape, KernelShape, typename InputShape::subtract<KernelShape>::add<NN::Shapes::unit<KernelShape::dimension>>, threads, ConvAdjuster>, NN::MapFunctions::Scale<WeightDatatype,DEPTH_VOLATILITY_FACTOR>>;

        template<typename Layer, typename Activation>
        using WithActivation = NN::Layers::Phantom::ActivationWrapper<Layer, Activation>;

        template<unsigned int ... vs>
        using Shape = NN::Shapes::Shape<vs...>;

        //define a bunch of Convoloution layers
        using a_Conv1 = WithActivation<BConv<Shape<28, 28>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using b_Conv1 = WithActivation<Conv<Shape<28, 28>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using c_Conv1 = WithActivation<Conv<Shape<28, 28>, Shape<3, 3>>, NN::Activations::LogisticActivation>;
        using a_Conv2 = WithActivation<Conv<Shape<26, 26>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using b_Conv2 = WithActivation<Conv<Shape<26, 26>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using c_Conv2 = WithActivation<Conv<Shape<26, 26>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;

        //define one with a larger kernel
        using d_Conv1 = WithActivation<Conv<Shape<28, 28>, Shape<5, 5>>, NN::Activations::LeakyReluActivation<WeightType>>;

        //define an early overlay layer (this should be a poor choice since we cant just overlay pixels over a bunch of weights and expect valuable information, but because we are summating, this can just go to zero)
        using e_Overlay1 = NN::Layers::Phantom::ActivationWrapper<NN::Layers::BiasedOverlay<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<28, 28>, NN::Shapes::Shape<24, 24>, threads, OverlayAdjuster>, NN::Activations::LogisticActivation>;

        //sequence together the earlier convs
        using a_ConvSeq = NN::NumberedSequence<a_Conv1, a_Conv2>;
        using b_ConvSeq = NN::NumberedSequence<b_Conv1, b_Conv2>;
        using c_ConvSeq = NN::NumberedSequence<c_Conv1, c_Conv2>;

        //Combine all of these
        using Concurrent_1 = NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, a_ConvSeq, b_ConvSeq, c_ConvSeq, e_Overlay1>;

        //define some more convoloution layers
        using f_Conv1 = WithActivation < Conv<Shape<24, 24>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using g_Conv1 = WithActivation<Conv<Shape<24, 24>, Shape<2, 2>>, NN::Activations::LeakyReluActivation<>>;
        using g_Conv2 = WithActivation<Conv<Shape<23, 23>, Shape<2, 2>>, NN::Activations::LeakyReluActivation<>>;
        
        //and another overlay
        using h_Overlay1 = WithActivation < NN::Layers::BiasedOverlay<WeightDatatype, ProcessingDevice, Shape<24, 24>, Shape<22, 22>, threads, OverlayAdjuster>, NN::Activations::LogisticActivation>;

        //sequence again
        using g_ConvSeq = NN::NumberedSequence<g_Conv1, g_Conv2>;

        //Combine again
        using Concurrent_2 = NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, f_Conv1, g_ConvSeq, h_Overlay1>;

        //define some more Convs
        using i_Conv1 = WithActivation<BConv<Shape<22, 22>, Shape<7, 7>>, NN::Activations::LeakyReluActivation<>>;
        using i_Conv2 = WithActivation<BConv<Shape<16, 16>, Shape<5, 5>>, NN::Activations::LeakyReluActivation<>>;

        //sequence
        using i_ConvSeq = NN::NumberedSequence<i_Conv1, i_Conv2>;

        //Test nested Concurrent sequences still work

        //define initial convs
        using j_Conv1 = WithActivation<Conv<Shape<22, 22>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using j_Conv2 = WithActivation < Conv<Shape<20, 20>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        //define some more convs to be run concurrently
        using j_3_a1 = WithActivation < BConv<Shape<18, 18>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using j_3_a2 = WithActivation < BConv<Shape<16, 16>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        //use a different kernel size, but still concurrently
        using j_3_b1 = WithActivation<BConv<Shape<18, 18>, Shape<5, 5>>, NN::Activations::LeakyReluActivation<>>;
        //use a conv and then an overlay
        using j_3_c1 = WithActivation < Conv<Shape<18, 18>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using j_3_c2 = WithActivation < NN::Layers::BiasedOverlay<WeightDatatype, ProcessingDevice, Shape<16, 16>, Shape< 14, 14>, threads, OverlayAdjuster>, NN::Activations::LeakyReluActivation<WeightType>>;
        //sequence all of the concurrent things
        using j_3_a_seq = NN::NumberedSequence<j_3_a1, j_3_a2>;
        using j_3_c_seq = NN::NumberedSequence<j_3_c1, j_3_c2>;

        //combine with DOT PRODUCT to test that reducer
        using j_3_1 = NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, j_3_a_seq, j_3_b1, j_3_c_seq>;
        //define a post-concurrent conv
        using j_3_2 = WithActivation<Conv<Shape<14, 14>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;

        //sequence the initial convs, concurrent sequence of various things, and post-concurrent convs
        using j_seq = NN::NumberedSequence<j_Conv1, j_Conv2, j_3_1, j_3_2>;

        //define some more convs
        using k_Conv1 = WithActivation<BConv<Shape<22, 22>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using k_Conv2 = WithActivation<BConv<Shape<20, 20>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using k_Conv3 = WithActivation<Conv<Shape<18, 18>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using k_Conv4 = WithActivation<Conv<Shape<16, 16>, Shape<3, 3>>, NN::Activations::LeakyReluActivation< >>;
        using k_Conv5 = WithActivation<Conv<Shape<14, 14>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using k_ConvSeq = NN::NumberedSequence<k_Conv1, k_Conv2, k_Conv3, k_Conv4, k_Conv5>;

        //again
        using l_Conv1 = WithActivation<BConv<Shape<22, 22>, Shape<3, 3>>, NN::Activations::LogisticActivation>;
        using l_Conv2 = WithActivation<BConv<Shape<20, 20>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<>>;
        using l_Conv3 = WithActivation<BConv<Shape<18, 18>, Shape<3, 3>>, NN::Activations::LogisticActivation>;
        //test Summation within a concurrent sequence as a reducer
        //define some dummy layers to combine
        using l_Conv4_1 = WithActivation<Conv<Shape<16, 16>, Shape<5, 5>>, NN::Activations::LogisticActivation>;
        using l_Conv4_2 = WithActivation<Conv<Shape<16, 16>, Shape<5, 5>>, NN::Activations::LeakyReluActivation<>>;
        using l_Overlay1 = WithActivation < NN::Layers::BiasedOverlay<WeightDatatype, ProcessingDevice, Shape<16, 16>, Shape< 12, 12>, threads, OverlayAdjuster>, NN::Activations::LeakyReluActivation<>>;
        //define the concurrent sequence
        using l_Concurrent = NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, l_Conv4_1, l_Conv4_2, l_Overlay1>;
        //define the sequence of l, which contains the concurrent
        using l_Seq = NN::NumberedSequence<l_Conv1, l_Conv2, l_Conv3, l_Concurrent>;

        //combine all the previous sequences, which contain concurrent sequences combined with DotProduct as well as Summation
        using Concurrent_3 = NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, i_ConvSeq, j_seq, k_ConvSeq, l_Seq>;
        
        //one more time but this time near the end
        //define a conv to 10 by 10
        using m_Conv1 = WithActivation<Conv<Shape<12, 12>, Shape<3, 3>>, NN::Activations::LeakyReluActivation<WeightType>>;
        //define an overlay from 10 by 10 to 1 by 10
        using CombinationLayer_Overlay = WithActivation<NN::Layers::BiasedOverlay<WeightDatatype, ProcessingDevice, NN::Shapes::Shape<10, 10>, NN::Shapes::Shape<1, 10>, threads, OverlayAdjuster>, NN::Activations::LeakyReluActivation<>>;
        //define a sequence of Convs from 10 by 10 to 1 by 10
        using CombinationLayer_Conv = NN::NumberedSequence<
            WithActivation<BConv<Shape<10,10>, Shape<3,1>>, NN::Activations::LeakyReluActivation<>>,
            WithActivation<Conv<Shape<8,10>, Shape<5,1>>, NN::Activations::LeakyReluActivation<>>,
            WithActivation<Conv<Shape<4, 10>, Shape<4, 1>>, NN::Activations::LeakyReluActivation<>>
        >;

        //define the Concurrent Combination, directly depositing the two results mlutitplied to the softmax layer, so they must both be correct and relevant
        using CombinationLayer = NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, CombinationLayer_Overlay, CombinationLayer_Conv>;

        //softmax
        using SoftmaxLayer = NN::Layers::Softmax<WeightDatatype, NN::Shapes::Shape<1, 10>, ProcessingDevice, threads>; //run softmax on the outputs

        //all of the above now need to be sequenced, for this we can use NN::NumberedSequence
        using Sequence = NN::NumberedSequence < CastLayer, Concurrent_1, Concurrent_2, Concurrent_3, m_Conv1, CombinationLayer, SoftmaxLayer> ; //runs each layer in sequence from left to right, passing the outputs from the previous layer as the inputs of the next layer without data copying


        //now wrap the overall network (in this case is just a single sequence) as a network and provide the error measure
        using Network = NN::Network<Sequence, NN::ErrorMeasures::CrossEntropyError>;
    }

    namespace MNIST_LARGE_TYPES {

        //An attempt at a decent architecture given the layers available at the time (Overlay, Convoloution, Softmax, ActivationWrapper, Cast, DeviceChange, Concurrent Sequence, [Numbered] Sequence)

        static constexpr bool USE_CUDA = true;
        static constexpr double DEPTH_VOLATILITY_FACTOR = 0.95; //by artificially deflating the errors of earlier layers we can provide a more stable search space for the later layers (because the earlier weights do not suddenly change by a relatively small amount that propogates to a large shift in values later)
        static constexpr unsigned int threads = 1;

        //define the dtype as it was defined during training - differs from AsDtype only by initialization values
        using WeightType = double;
        using WeightDatatype = NN::Dtypes::AsDtypeScaledWithOffset<WeightType, 0.0001, 0.005>; // initialize weights as 0.01 +- 0.0005*10
        using InputDatatype = NN::Dtypes::AsDtype<uint8_t>; //we receive a 28x28 grid of values between 0 and 255

        //define the device to process on (the network is auto-adjusted to start and end on CPU)
        using ProcessingDevice = std::conditional_t<USE_CUDA, NN::Devices::CUDA<0>, NN::Devices::CPU>; //determines the device based on the USE_CUDA constexpr

        using ConvAdjuster = NN::WeightModifiers::ClampedLinear<double, 0.002, 0.000008>; //define adjuster for convoloution layers
        using OverlayAdjuster = NN::WeightModifiers::ClampedLinear<double, 0.005, 0.000008>; //define adjuster for overlay layers

        using CastLayer = NN::Layers::Cast<InputDatatype, WeightDatatype, NN::Shapes::Shape<28, 28>, ProcessingDevice, 1>; //define a casting layer from the input type of 0-255 uints to the type of the weights (double)

        //define shorthand template alias
        template<typename InputShape, typename KernelShape>
        using BConv = NN::Layers::Phantom::ErrorPreprocessor< //make early layers less volatile
            NN::Layers::BiasedConvoloution < WeightDatatype, ProcessingDevice, InputShape, KernelShape, typename InputShape::subtract<KernelShape>::add<NN::Shapes::unit<KernelShape::dimension>>, threads, ConvAdjuster>,
            NN::MapFunctions::Scale<WeightDatatype, DEPTH_VOLATILITY_FACTOR>>;


        template<typename InputShape, typename OutputShape>
        using BOver = NN::Layers::Phantom::ErrorPreprocessor< //make early layers less volatile
            NN::Layers::BiasedOverlay < WeightDatatype, ProcessingDevice, InputShape,OutputShape, threads, OverlayAdjuster>,
            NN::MapFunctions::Scale<WeightDatatype, DEPTH_VOLATILITY_FACTOR>>;


        template<typename InputShape, typename KernelShape>
        using Conv = NN::Layers::Phantom::ErrorPreprocessor<
            NN::Layers::BiasedConvoloution < WeightDatatype, ProcessingDevice, InputShape, KernelShape, typename InputShape::subtract<KernelShape>::add<NN::Shapes::unit<KernelShape::dimension>>, threads, ConvAdjuster>,
            NN::MapFunctions::Scale<WeightDatatype, DEPTH_VOLATILITY_FACTOR>>;

        template<typename Layer, typename Activation>
        using WithActivation = NN::Layers::Phantom::ActivationWrapper<Layer, Activation>;

        template<typename Layer>
        using Leaky = WithActivation<Layer, NN::Activations::LeakyReluActivation<>>;

        template<typename Layer>
        using Logistic = WithActivation<Layer, NN::Activations::LogisticActivation>;

        //shorthand for NN::Shapes::Shape
        template<unsigned int ... vs>
        using Shape = NN::Shapes::Shape<vs...>;


        //we can also define everything in-place rather than aliasing:

        using Layers = NN::NumberedSequence<
            Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, //28x28
                NN::NumberedSequence<
                    Leaky<BConv<Shape<28,28>, Shape<3,3>>>,
                    Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, //26x26
                           NN::NumberedSequence<
                                Leaky<BConv<Shape<26,26>, Shape<3,3>>>,
                                Leaky<BConv<Shape<24,24>, Shape<3,3>>>
                           >,
                           Logistic<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation,
                                Leaky<BOver<Shape<26,26>, Shape<22,22>>>,
                                Leaky<BConv<Shape<26,26>, Shape<5,5>>>
                            >>,
                            NN::NumberedSequence<
                                Leaky<BConv<Shape<26,26>,Shape<4,4>>>,
                                Leaky<BConv<Shape<23,23>,Shape<2,2>>>
                            >
                    >>,
                    Leaky<BConv<Shape<22,22>,Shape<3,3>>>
                >,
                NN::NumberedSequence<
                    NN::Sequences::ConcurrentSequence<NN::Reducers::Summation,
                        NN::NumberedSequence<
                            Conv<Shape<28,28>,Shape<3,3>>,
                            Leaky<BOver<Shape<26,26>,Shape<20,20>>>
                        >, 
                        NN::NumberedSequence<
                            Conv<Shape<28,28>,Shape<5,5>>,
                            Leaky<BOver<Shape<24,24>,Shape<20,20>>>
                        >,
                        NN::NumberedSequence<
                            Conv<Shape<28,28>,Shape<7,7>>,
                            Leaky<BOver<Shape<22,22>,Shape<20,20>>>
                        >, 
                        NN::NumberedSequence<
                            Conv<Shape<28,28>,Shape<7,7>>,
                            Logistic<BOver<Shape<22,22>,Shape<20,20>>>
                        >,
                        NN::NumberedSequence<
                            Logistic<Conv<Shape<28,28>,Shape<7,7>>>,
                            Leaky<BOver<Shape<22,22>,Shape<20,20>>>
                        >
                    >
                >,
                NN::NumberedSequence<
                    Leaky<Conv<Shape<28,28>, Shape<5,5>>>,
                    Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, //24x24
                       NN::NumberedSequence<
                            Leaky<BConv<Shape<24,24>,Shape<3,3>>>,
                            Leaky<BConv<Shape<22,22>,Shape<3,3>>>
                       >,
                      Leaky<BConv<Shape<24,24>,Shape<5,5>>>,
                      Leaky<BOver<Shape<24,24>,Shape<20,20>>>,
                      Logistic<BOver<Shape<24,24>,Shape<20,20>>>
                 >>
                >
           >>,
           Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, //20x20    
                NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, //20x20
                       NN::NumberedSequence<
                            Leaky<BConv<Shape<20,20>, Shape<4,4>>>,
                            Leaky<BConv<Shape<17,17>, Shape<2,2>>>
                       >,
                       NN::NumberedSequence<
                           Leaky<Conv<Shape<20,20>,Shape<3,3>>>,
                           Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, //18x18
                                Leaky<BConv<Shape<18,18>, Shape<3,3>>>,
                                Logistic<BConv<Shape<18, 18>, Shape<3, 3>>>
                             >>
                      >
                 >,
                 Leaky<BConv<Shape<20,20>,Shape<5,5>>>,
                 Leaky<BOver<Shape<20,20>,Shape<16,16>>>,
                 NN::NumberedSequence<
                    Leaky<BConv<Shape<20,20>,Shape<3,3>>>,
                    Leaky<BOver<Shape<18,18>,Shape<16,16>>>
                 >
           >>,
           Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, //16x16
                NN::NumberedSequence<
                       NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, //16x16
                            Leaky<BConv<Shape<16,16>,Shape<5,5>>>,
                            Logistic<BConv<Shape<16, 16>, Shape<5, 5>>>
                        >,
                        Leaky<BConv<Shape<12,12>,Shape<3,3>>>
                >,
                NN::NumberedSequence<
                    Leaky<Conv<Shape<16,16>,Shape<3,3>>>,
                    Leaky<BConv<Shape<14,14>,Shape<2,2>>>,
                    Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, //13x13
                        NN::NumberedSequence<
                            Leaky<BConv<Shape<13,13>,Shape<2,2>>>,
                            Leaky<BConv<Shape<12,12>, Shape<2,2>>>,
                            Leaky<BConv<Shape<11,11>,Shape<2,2>>>
                        >,
                        NN::NumberedSequence<
                            Leaky<BConv<Shape<13,13>,Shape<2,2>>>,
                            Leaky<BConv<Shape<12,12>,Shape<3,3>>>
                        >,
                        NN::NumberedSequence<
                            Leaky<BConv<Shape<13,13>,Shape<3,3>>>,
                            Leaky<BConv<Shape<11,11>,Shape<2,2>>>
                        >,
                        Leaky<BConv<Shape<13,13>,Shape<4,4>>>,
                        Leaky<BOver<Shape<13,13>,Shape<10,10>>>
                    >>
                >,
                Leaky<BOver<Shape<16,16>,Shape<10,10>>>
            >>,
            Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::Summation, //10x10
                BConv<Shape<10,10>,Shape<6,6>>,
                BOver<Shape<10,10>,Shape<5,5>>,
                NN::NumberedSequence<
                    BConv<Shape<10,10>,Shape<3,3>>,
                    BConv<Shape<8,8>,Shape<3,3>>,
                    BConv<Shape<6,6>,Shape<2,2>>
                >,
                Leaky<NN::Sequences::ConcurrentSequence<NN::Reducers::DotProduct, //10x10
                       NN::NumberedSequence<
                            Leaky<BConv<Shape<10,10>,Shape<4,4>>>,
                            Logistic<BConv<Shape<7,7>,Shape<3,3>>>
                       >,
                       Leaky<BConv<Shape<10,10>,Shape<6,6>>>
                >>
            >>,
            
        >;


                


        using SoftmaxLayer = NN::Layers::Softmax<WeightDatatype, NN::Shapes::Shape<1,10>, ProcessingDevice, threads>; //run softmax on the outputs

        //all of the above now need to be sequenced, for this we can use NN::NumberedSequence
        using Sequence = NN::NumberedSequence < CastLayer, Layers, Leaky<BOver<Shape<5,5>,Shape<1,10>>>, SoftmaxLayer>; //runs each layer in sequence from left to right, passing the outputs from the previous layer as the inputs of the next layer without data copying


        //now wrap the overall network (in this case is just a single sequence) as a network and provide the error measure
        using Network = NN::Network<Sequence, NN::ErrorMeasures::CrossEntropyError>;
    }

    namespace MNIST_XL_TYPES {

       


        using Sequence = NN::NumberedSequence < MNIST_LARGE_TYPES::CastLayer, MNIST_LARGE_TYPES::Layers, MNIST_LARGE_TYPES::Leaky<MNIST_LARGE_TYPES::BOver<NN::Shapes::Shape<5, 5>, NN::Shapes::Shape<28, 28>>>, MNIST_LARGE_TYPES::Layers, MNIST_LARGE_TYPES::Leaky<MNIST_LARGE_TYPES::BOver<NN::Shapes::Shape<5, 5>, NN::Shapes::Shape<1, 10>>>, MNIST_LARGE_TYPES::SoftmaxLayer>; //runs each layer in sequence from left to right, passing the outputs from the previous layer as the inputs of the next layer without data copying


        //now wrap the overall network (in this case is just a single sequence) as a network and provide the error measure
        using Network = NN::Network<Sequence, NN::ErrorMeasures::CrossEntropyError>;
    }

    /// <summary>
    /// A MNIST dataset network type with 2499 parameters capable of achieving just over 80% on the testing dataset (trained only on the training dataset)
    /// </summary>
    using MNIST_SMALL = MNIST_SMALL_TYPES::Network;

    using MNIST_MEDIUM = MNIST_MEDIUM_TYPES::Network;

    using MNIST_LARGE = MNIST_LARGE_TYPES::Network;

    using MNIST_XL = MNIST_XL_TYPES::Network;
}