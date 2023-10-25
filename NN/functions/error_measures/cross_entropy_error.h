#pragma once
#include <cuda_runtime.h>
#include "../../helpers/string_literal.h"
namespace NN {
    namespace ErrorMeasures {

        //example macro definition
        struct CrossEntropyError final {
            static constexpr StringLiteral name = StringLiteral("CrossEntropy");
            template<typename T>
            __host__ __device__ static T apply(const T output, const T expected) {
                return static_cast<T>(- (log(output + 0.0000001)) * expected);
            }
            template<typename T>
            __host__ __device__ static T derivative(const T output, const T expected) {
                    return static_cast<T>(- (expected / (output + 0.0000001)));
            } 

            template<typename T>
            __host__ __device__ static T average(T const *const outputs, T const *const expected, const unsigned int size) {
                T sum = 0;
                for (unsigned int i = 0; i < size; i++) {
                    sum += apply(outputs[i], expected[i]) / size;
                } return sum;
            }
        };
    }
}