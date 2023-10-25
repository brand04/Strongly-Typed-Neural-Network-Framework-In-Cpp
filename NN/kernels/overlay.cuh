#pragma once
#include <cuda_runtime.h>
#include "../references/includes.h"
#include "../devices/includes.h"
#include "../dtypes/concept.h"
#include "../shapes/includes.h"
#include "../cuda_helpers/kernel_launch.h"
#include "../threads/thread_context.h"
#include "../asserts/assert_t.h"
#include "../asserts/false_assert.h"

//Abstract away the specifics of which device to launch the kernel on through these methods using if constexpr (...) flow control
namespace NN {
	namespace Kernels {
		namespace Overlay {

		

            namespace Definitions {
                //the function for the kernel

                //defines cuda and cpu function
                template<Dtypes::Dtype Dtype, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape, Shapes::IsShape BiasShape>
                __host__ __device__ inline void slidingKernelImpl(
                    typename Dtype::Type* const output,
                    typename Dtype::Type const* const kernel,
                    typename Dtype::Type const* const matrix,
                    typename Dtype::Type const* const biasLocation,
                    const unsigned int& id
                ) {
                    const Shapes::RuntimeShape<OutputShape::dimension> offset = OutputShape::runtimeUnflatten(id); //since id is not known at compile-time, assign it to a RuntimeShape

                    typename Dtype::Type sum = Dtype::additiveIdentity;

                    //compute the value of kernel and matrix at each of the positions within the offsetted kernel and multiply them together before summing
                    //for each possible position within the kernel
                    for (unsigned int i = 0; i < KernelShape::volume; i++) {
                        sum += kernel[i] * matrix[MatrixShape::runtimeFlatten(offset.add(KernelShape::runtimeUnflatten(i)))]; //multiply that kernel position with the matrix value at (offset+kernelPosition)
                    }
                    if constexpr (!std::same_as<Shapes::zero<1>, BiasShape>) sum += biasLocation[id]; //every output has a unique bias

                    output[id] = sum;
                }



                template <Dtypes::Dtype Dtype, typename W, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape, Shapes::IsShape BiasShape>
                __host__ __device__ inline void adjustMatrixWeightsSlidingKernelImpl(
                    typename Dtype::Type const* const nodeDeltas,
                    typename Dtype::Type const* const kernel,
                    typename Dtype::Type* const matrix,
                    typename Dtype::Type* const biases,
                    const unsigned int id
                ) {
                    //create shapes of the highest valid index rather than size
                    using MatrixShapeMinus = MatrixShape::subtract<Shapes::unit<MatrixShape::dimension>>;
                    using OutputShapeMinus = OutputShape::subtract<Shapes::unit<OutputShape::dimension>>;
                    using KernelShapeMinus = KernelShape::subtract<Shapes::unit<KernelShape::dimension>>;
                    //underlying dtype
                    using dtype = typename Dtype::Type;

                    if (id < MatrixShape::volume) { //handling a matrix weight
                        const Shapes::RuntimeShape<OutputShape::dimension> matrixOffset = MatrixShape::runtimeUnflatten(id);
                        //matrixOffset contains coordinates of the matrix value to adjust

                        dtype sum = Dtype::additiveIdentity;

                        //compute first valid position for each dimension - all valid kernel positions will have values greater than this
                        Shapes::RuntimeShape<KernelShape::dimension> initialKernelOffset {};
                        Shapes::RuntimeShape<KernelShape::dimension> finalKernelOffset {};
                        Shapes::RuntimeShape<OutputShape::dimension> initialOutputOffset {};
                        Shapes::RuntimeShape<OutputShape::dimension> finalOutputOffset {};
                        for (unsigned int i = 0; i < KernelShape::dimension; i++) {
                            //earliest component of this dimension that will cause a valid kernelOffset configuration - check for overflow at the end of the matrix
                            //check whether the matrixOffset component will be greater than the edge of the matrix once the greatest kernel added - which translates to check if matrixOffset component is greater than the OutputShape and allow starting from 0 kernelOffset
                            //initialKernelOffset.components[i] =  matrixOffset.components[i] > OutputShapeMinus::asRuntimeShape().components[i] ? matrixOffset.components[i] : 0;
                            initialKernelOffset.components[i] = matrixOffset.components[i] > OutputShapeMinus::asRuntimeShape().components[i] ? matrixOffset.components[i] - OutputShapeMinus::asRuntimeShape().components[i] : 0;
                            //lastest component of this dimension that will cause a valid kernelOffset configuration
                            //check whether the matrixOffset component will be greater than the edge of the matrix once the greatest kernel added - which translates to check if matrixOffset component is greater than the OutputShape and allow starting from 0 kernelOffset
                            finalKernelOffset.components[i] = matrixOffset.components[i] < KernelShapeMinus::asRuntimeShape().components[i] ? matrixOffset.components[i] : KernelShapeMinus::asRuntimeShape().components[i];

                            //kernelShapeMinus?
                            initialOutputOffset.components[i] = (matrixOffset.components[i] - initialKernelOffset.components[i]);
                            finalOutputOffset.components[i] = (matrixOffset.components[i] - finalKernelOffset.components[i]);
                        }
                        Shapes::RuntimeShape<KernelShape::dimension> kernelOffset = initialKernelOffset;
                        Shapes::RuntimeShape<KernelShape::dimension> _ = KernelShapeMinus::asRuntimeShape();

                        unsigned int kernelPosition = KernelShape::runtimeFlatten(initialKernelOffset); //position within kernel initially
                        unsigned int outputPosition = OutputShape::runtimeFlatten(initialOutputOffset);

                        unsigned int d = KernelShape::dimension - 1;
                        bool exit = false;
                        while (!exit) {

                            sum += kernel[kernelPosition] * nodeDeltas[outputPosition];



                            while (kernelOffset.components[d] >= finalKernelOffset.components[d]) {
                                if (d == 0) {
                                    exit = true; //exit point
                                    break;
                                }

                                else d--; //reduce d because this dimennsion cannot be incremented anymore



                            }





                            if (!exit) { //skip if we are gonna exit 

                                kernelOffset.components[d]++;


                                if constexpr (KernelShape::dimension > 1) {

                                    //kernelPosition += (d == KernelShape::dimension-1) ? 1 : KernelShape::partialFactors::asRuntimeShape().components[d];
                                    //outputPosition -= (d == OutputShape::dimension-1) ? 1 : OutputShape::partialFactors::asRuntimeShape().components[d];
                                    kernelPosition++;
                                    outputPosition--;
                                    ++d; //handled this dimension, the rest should be changed back to initialKernelOffset

                                    //reincrement d to the lowest dimension
                                    for (; d < KernelShape::dimension; d++) {
                                        kernelOffset.components[d] = initialKernelOffset.components[d];

                                        kernelPosition += ((d == KernelShape::dimension - 1) ? 1 : KernelShape::partialFactors::asRuntimeShape().components[d]) *
                                            (initialKernelOffset.components[d] + (KernelShapeMinus::asRuntimeShape().components[d] - finalKernelOffset.components[d]));


                                        outputPosition -= ((d == OutputShape::dimension - 1) ? 1 : OutputShape::partialFactors::asRuntimeShape().components[d]) *
                                            (finalOutputOffset.components[d] + (OutputShapeMinus::asRuntimeShape().components[d] - initialOutputOffset.components[d]));



                                    }
                                    --d;
                                }
                                else {
                                    kernelPosition += 1;
                                    outputPosition -= 1;
                                }
                            }




                        }


                        //dtype test = W::adjust(matrix[id], sum);

                        W::adjust(matrix[id], sum);
                    }
                    else { //handling a bias weight
                        const unsigned int biasId = id - MatrixShape::volume;
                        W::adjust(biases[biasId], nodeDeltas[biasId]); // bias only impacts a single output node, therefore the node delta is equal to the error for the bias
                    }

                }

                template <Dtypes::Dtype Dtype, typename KernelShape, typename OutputShape, typename MatrixShape>
                __host__ __device__ void backpropKernelNodeDeltasImpl(
                    typename Dtype::Type* const preNodeDeltas,
                    typename Dtype::Type const* const postNodeDeltas,
                    typename Dtype::Type const* const matrix,
                    const unsigned int id
                ) {
                    using dtype = typename Dtype::Type;


                    Shapes::RuntimeShape<OutputShape::dimension> kernelOffset = KernelShape::runtimeUnflatten(id);


                    //for every output
                    dtype sum = Dtype::additiveIdentity;
                    for (int i = 0; i < OutputShape::volume; i++) {
                        sum += postNodeDeltas[i] * matrix[MatrixShape::runtimeFlatten(kernelOffset.add(OutputShape::runtimeUnflatten(i)))]; // error * weight (where the weight location is calculated from the kernelOffset and outputOffset)
                    }
                    preNodeDeltas[id] = sum;

                }


                //entry point to cuda device - less strongly typed
                template<Dtypes::Dtype Dtype, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape, Shapes::IsShape BiasShape>
                __global__ void slidingKernelLaunch(
                    typename Dtype::Type* const output,
                    typename Dtype::Type const* const kernel,
                    typename Dtype::Type const* const matrix,
                    typename Dtype::Type const* const biasLocation
                ) {
                    const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
                    if (id < OutputShape::volume) { //no overflow even when extra blocks assigned
                        slidingKernelImpl<Dtype, KernelShape, OutputShape, MatrixShape, BiasShape>(output, kernel, matrix, biasLocation, id);
                    }
                }




                //entry point to cuda device - less strongly typed
                template<Dtypes::Dtype Dtype, typename W, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape, Shapes::IsShape BiasShape>
                __global__ void adjustWeightsKernelLaunch(
                    typename Dtype::Type const* const nodeDeltas,
                    typename Dtype::Type const* const kernel,
                    typename Dtype::Type *const matrix,
                    typename Dtype::Type *const biasLocation
                ) {
                    static_assert(std::same_as<BiasShape, Shapes::zero<1>> || std::same_as<OutputShape, BiasShape>, "Support for Biases not of shape OutputShape is not available currently");
                    const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
                    if (id < MatrixShape::volume + BiasShape::volume) { //no overflow even when extra blocks assigned
                        adjustMatrixWeightsSlidingKernelImpl<Dtype, W, KernelShape, OutputShape, MatrixShape, BiasShape>(nodeDeltas, kernel, matrix, biasLocation, id);
                    }
                }

                template<Dtypes::Dtype Dtype, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape>
                __global__ void backpropKernelNodeDeltasLaunch(
                    typename Dtype::Type* const preNodeDeltas,
                    typename Dtype::Type const* const postNodeDeltas,
                    typename Dtype::Type const* const matrix
                ) {
                    const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
                    if (id < KernelShape::volume) {
                        //avoid overflow if extra threads assigned
                        backpropKernelNodeDeltasImpl<Dtype, KernelShape, OutputShape, MatrixShape>(preNodeDeltas, postNodeDeltas, matrix, id);
                    }
                }
            }

            template<Dtypes::Dtype Dtype, Devices::Device Device, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape, Shapes::IsShape BiasShape>
            inline void slidingKernel(References::MutableShapedDeviceReference<typename Dtype::Type, OutputShape, Device> output,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, KernelShape, Device> kernel,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, MatrixShape, Device> matrix,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, BiasShape, Device> biasLocation,
                Threads::ThreadContext& ctx
            ) {


                if constexpr (Devices::CUDADevice<Device>) {
                    //Cuda path

                    Definitions::slidingKernelLaunch<Dtype, KernelShape, OutputShape, MatrixShape, BiasShape> << <CudaHelpers::computeGridSize(OutputShape::volume), CudaHelpers::computeBlockSize(OutputShape::volume), 0, ctx.stream >> > (output, kernel, matrix, biasLocation);


                }
                else if constexpr (Devices::CPUDevice<Device>) {
                    //CPU path - perform in sequence
                    for (unsigned int id = 0; id < OutputShape::volume; id++) {
                        Definitions::slidingKernelImpl<Dtype,KernelShape,OutputShape,MatrixShape,BiasShape>(output, kernel, matrix, biasLocation, id);
                    }
                }
                else {
                    static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Invalid Device - only CUDA and CPU are supported");
                }
            }
            
            

            template<Dtypes::Dtype Dtype, typename W, Devices::Device Device, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape, Shapes::IsShape BiasShape>
            inline void adjustWeightsKernel(
                References::ImmutableShapedDeviceReference<typename Dtype::Type, OutputShape, Device> nodeDeltas,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, KernelShape, Device> kernel,
                References::MutableShapedDeviceReference<typename Dtype::Type, MatrixShape, Device> matrix,
                References::MutableShapedDeviceReference<typename Dtype::Type, BiasShape, Device> biasLocation,
                Threads::ThreadContext& ctx
            ) {


                if constexpr (Devices::CUDADevice<Device>) {
                    //Cuda path

                    Definitions::adjustWeightsKernelLaunch<Dtype, W,KernelShape, OutputShape, MatrixShape, BiasShape> << <CudaHelpers::computeGridSize(MatrixShape::volume + BiasShape::volume), CudaHelpers::computeBlockSize(MatrixShape::volume + BiasShape::volume), 0, ctx.stream >> > (nodeDeltas, kernel, matrix, biasLocation);


                }
                else if constexpr (Devices::CPUDevice<Device>) {
                    //CPU path - perform in sequence
                    for (unsigned int id = 0; id < MatrixShape::volume + BiasShape::volume; id++) {
                        Definitions::adjustMatrixWeightsSlidingKernelImpl<Dtype,W, KernelShape, OutputShape, MatrixShape, BiasShape>(nodeDeltas, kernel, matrix, biasLocation, id);
                    }
                }
                else {
                    static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Invalid Device - only CUDA and CPU are supported");
                }
            }
           
            /// <summary>
            /// Backpropogates Deltas from outputs to inputs for an overlay layer whose weights are the matrix
            /// </summary>
            /// <typeparam name="Dtype">The Dtype of the involved values</typeparam>
            /// <typeparam name="Device">The device on which the data is stored and on which the data will be processed</typeparam>
            /// <typeparam name="KernelShape">The shape of the Kernel (which in this case is the InputShape)</typeparam>
            /// <typeparam name="OutputShape">The shape of the Output</typeparam>
            /// <typeparam name="MatrixShape">The shape of the Matrix</typeparam>
            /// <param name="preNodeDeltas">The deltas for the inputs</param>
            /// <param name="postNodeDeltas">The deltas for the outputs</param>
            /// <param name="matrix">The matrix weights (no need for bias)</param>
            /// <param name="ctx">The context for the thread</param>
            template<Dtypes::Dtype Dtype, Devices::Device Device, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape>
            inline void backpropDeltasKernel(
                References::MutableShapedDeviceReference<typename Dtype::Type, KernelShape, Device> preNodeDeltas,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, OutputShape, Device> postNodeDeltas,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, MatrixShape, Device> matrix,
                Threads::ThreadContext& ctx
            ) {


                if constexpr (Devices::CUDADevice<Device>) {
                    //Cuda path

                    Definitions::backpropKernelNodeDeltasLaunch<Dtype, KernelShape, OutputShape, MatrixShape> << <CudaHelpers::computeGridSize(KernelShape::volume), CudaHelpers::computeBlockSize(KernelShape::volume), 0, ctx.stream >> > (preNodeDeltas, postNodeDeltas, matrix);


                }
                else if constexpr (Devices::CPUDevice<Device>) {
                    //CPU path - perform in sequence
                    for (unsigned int id = 0; id < KernelShape::volume; id++) {
                        Definitions::backpropKernelNodeDeltasImpl<Dtype,KernelShape,OutputShape,MatrixShape>(preNodeDeltas, postNodeDeltas, matrix, id);
                    }
                }
                else {
                    static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Invalid Device - only CUDA and CPU are supported");
                }
            }


		}
	}
}