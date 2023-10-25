#pragma once

#include "../devices/includes.h"
#include "../dtypes/concept.h"
#include "../shapes/includes.h"
#include "../cuda_helpers/kernel_launch.h"
#include "../threads/thread_context.h"
#include "../asserts/assert_t.h"
#include "../asserts/false_assert.h"

#include "./overlay.cuh" //uses same slidingKernel
namespace NN {

	namespace Kernels {

		namespace Convoloution {

			namespace Definitions {

                template <Dtypes::Dtype Dtype, Shapes::IsShape KernelShape, Shapes::IsShape OutputShape, Shapes::IsShape MatrixShape>
                __host__ __device__ inline void backpropMatrixDeltasImpl(
                    typename Dtype::Type* const preNodeDeltas,
                    typename Dtype::Type const * const postNodeDeltas,
                    typename Dtype::Type const* const kernel,
                    const unsigned int id
                ) {
                    //create shapes of the highest valid index rather than size
                    using MatrixShapeMinus = MatrixShape::subtract<Shapes::unit<MatrixShape::dimension>>;
                    using OutputShapeMinus = OutputShape::subtract<Shapes::unit<OutputShape::dimension>>;
                    using KernelShapeMinus = KernelShape::subtract<Shapes::unit<KernelShape::dimension>>;
                    //underlying dtype
                    using dtype = typename Dtype::Type;


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

                        sum += kernel[kernelPosition] * postNodeDeltas[outputPosition];



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

                    preNodeDeltas[id] = sum;

                }

                template <Dtypes::Dtype Dtype, typename W, typename KernelShape, typename OutputShape, typename MatrixShape, typename BiasShape>
                __host__ __device__ inline void adjustKernelWeightsImpl(
                    typename Dtype::Type* const kernel,
                    typename Dtype::Type* const biases,
                    typename Dtype::Type const* const postNodeDeltas,
                    typename Dtype::Type const* const matrix,
                    const unsigned int id
                ) {
                    using dtype = typename Dtype::Type;

                    if (id < KernelShape::volume) {
                        Shapes::RuntimeShape<OutputShape::dimension> kernelOffset = KernelShape::runtimeUnflatten(id);


                        //for every output
                        dtype sum = Dtype::additiveIdentity;
                        for (int i = 0; i < OutputShape::volume; i++) {
                            sum += postNodeDeltas[i] * matrix[MatrixShape::runtimeFlatten(kernelOffset.add(OutputShape::runtimeUnflatten(i)))]; // error * weight (where the weight location is calculated from the kernelOffset and outputOffset)
                        }
                        W::adjust(kernel[id], sum);
                    }
                    else {
                        W::adjust(biases[id - KernelShape::volume], postNodeDeltas[id - KernelShape::volume]);
                    }

                }

                template <Dtypes::Dtype Dtype, typename W, typename KernelShape, typename OutputShape, typename MatrixShape, typename BiasShape>
                __global__ void adjustKernelWeightsLaunch(
                    typename Dtype::Type* const kernel,
                    typename Dtype::Type* const biases,
                    typename Dtype::Type const* const postNodeDeltas,
                    typename Dtype::Type const* const inputs
                ) {

                    const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
                    if (id < KernelShape::volume + BiasShape::volume) { //no overflow even when extra blocks assigned
                        adjustKernelWeightsImpl<Dtype, W, KernelShape, OutputShape, MatrixShape, BiasShape>(kernel, biases, postNodeDeltas, inputs, id);
                    }
                }

                template<Dtypes::Dtype Dtype, typename KernelShape, typename OutputShape, typename MatrixShape>
                __global__ void backpropMatrixDeltasLaunch(
                    typename Dtype::Type* const preNodeDeltas,
                    typename Dtype::Type const* const postNodeDeltas,
                    typename Dtype::Type const* const kernel
                ) {
                    const unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
                    if (id < MatrixShape::volume) {
                        backpropMatrixDeltasImpl<Dtype, KernelShape, OutputShape, MatrixShape>(preNodeDeltas, postNodeDeltas, kernel, id);
                    }
                }
			}

            template<Dtypes::Dtype Dtype, typename W, typename KernelShape, typename BiasShape, typename OutputShape, typename MatrixShape, typename Device>
            inline void adjustKernelWeights(
                References::MutableShapedDeviceReference<typename Dtype::Type, KernelShape, Device> kernel,
                References::MutableShapedDeviceReference<typename Dtype::Type, BiasShape, Device> biases,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, MatrixShape, Device> inputs,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, OutputShape, Device> postNodeDeltas,
                Threads::ThreadContext& ctx
            ) {

                if constexpr (Devices::CUDADevice<Device>) {
                    Definitions::adjustKernelWeightsLaunch<Dtype, W, KernelShape, OutputShape, MatrixShape, BiasShape> << <CudaHelpers::computeGridSize(KernelShape::volume + BiasShape::volume), CudaHelpers::computeBlockSize(KernelShape::volume + BiasShape::volume), 0, ctx.stream >> > (kernel, biases, postNodeDeltas, inputs);
                }
                else if constexpr (Devices::CPUDevice<Device>) {
                    for (unsigned int id = 0; id < KernelShape::volume + BiasShape::volume; id++) {
                        Definitions::adjustKernelWeightsImpl<Dtype, W, KernelShape, OutputShape, MatrixShape, BiasShape>(kernel, biases, postNodeDeltas, inputs, id);
                    }
                }
                else {
                    static_assert(struct_assert<Asserts::AssertFalseWith<Device>>, "Only CPU and CUDA are supported");
                }
            }

            template<Dtypes::Dtype Dtype, typename KernelShape, typename MatrixShape, typename OutputShape, typename Device>
            inline void backpropMatrixDeltas(
                References::MutableShapedDeviceReference<typename Dtype::Type, MatrixShape, Device> preNodeDeltas,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, OutputShape, Device> postNodeDeltas,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, KernelShape, Device> kernel,
                Threads::ThreadContext& ctx
            ) {

                if constexpr (Devices::CUDADevice<Device>) {
                    Definitions::backpropMatrixDeltasLaunch<Dtype, KernelShape, OutputShape, MatrixShape> << <CudaHelpers::computeGridSize(MatrixShape::volume), CudaHelpers::computeBlockSize(MatrixShape::volume), 0, ctx.stream >> > (preNodeDeltas, postNodeDeltas, kernel);
                }
                else if constexpr (Devices::CPUDevice<Device>) {
                    for (unsigned int id = 0; id < MatrixShape::volume; id++) {
                        Definitions::backpropMatrixDeltasImpl<Dtype, KernelShape, OutputShape, MatrixShape>(preNodeDeltas, postNodeDeltas, kernel, id);
                    }
                }
                else {
                    static_assert(struct_assert < Asserts::AssertFalseWith<Device>>, "Only CUDA and CPU are supported");
                }
            }

            template<Dtypes::Dtype Dtype, typename KernelShape, typename BiasShape, typename OutputShape, typename MatrixShape, typename Device>
            inline void convoloution(
                References::MutableShapedDeviceReference<typename Dtype::Type, OutputShape, Device> output,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, MatrixShape, Device> input,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, KernelShape, Device> kernel,
                References::ImmutableShapedDeviceReference<typename Dtype::Type, BiasShape, Device> biases,
                Threads::ThreadContext& ctx
            ) {
                Overlay::slidingKernel<Dtype>(output, kernel, input, biases, ctx); //same as overlay but the input is the matrix rather than kernel
            }
		}
	}
 }