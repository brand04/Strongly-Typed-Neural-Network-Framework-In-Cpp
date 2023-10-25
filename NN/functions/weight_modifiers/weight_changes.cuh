/*
Defines how weights are adjusted, based upon the weights previous value and the change that the backpropogation algorithm has calculated
For example, this means that weights could be changes in a logarithmic way, or if they are above a threshold, changed by a greater amount

*/


#define _WEIGHT_CHANGE_DEVICES __host__ __device__
#define _T template<typename T>
#define _WEIGHT_CHANGE_DECLARATION(TYPE) _T struct TYPE : WeightFunction<T> { \
		_WEIGHT_CHANGE_DEVICES T adjust(T current, T delta) override; \
} \


#define _WEIGHT_CHANGE_FUNCTION(TYPE) _T _WEIGHT_CHANGE_DEVICES T TYPE<T>::adjust(T current, T delta) \



#pragma once

namespace WeightAdjusters {


	/*defines how weights are changed
	*
	*  The struct is specific for each layer, created upon the construction of the layer and thus cannot be swapped once created. However additional data can be included into the struct, allowing for many possible ways to adjust weights
	*  The result of 'adjust' is SUBTRACTED from the value of the weight
	*  Also note that if cuda assisted weight changing is enabled, the properties will only ever be that for which they were constructed, and will not change over time
	*/
	template <typename T>
	struct WeightFunction {
		_WEIGHT_CHANGE_DEVICES WeightFunction() {}
		_WEIGHT_CHANGE_DEVICES virtual T adjust(T current, T delta) = 0;
	};











	//example declaration (LinearChange) - no modification to delta
	template <typename T, T LearningRate = (T)0.0001>
	struct LinearChange : WeightFunction<T> {
		_WEIGHT_CHANGE_DEVICES T adjust(T current, T delta) override {
			return LearningRate * delta;
		}
	};


	//example declaration with additional parameters
	template <typename T, T Threshold = (T)1000, T LearningRate = 0.01>
	struct ClampedLinearChange : LinearChange<T> {
		
		_WEIGHT_CHANGE_DEVICES T adjust(T current, T delta) override {
			if (LearningRate * delta > Threshold) return Threshold;
			if (LearningRate * delta < -Threshold) return -Threshold;
			else return LearningRate * delta;
		}
	};


	/*
	 shorthand declarations, as no extra detail is needed.Automatically inherits from WeightFunctionand copies the constructor.
	 This can be changes if a different constructor is desired or if additional properties are wanted
	*/
	_WEIGHT_CHANGE_DECLARATION(LogisticChange);
	_WEIGHT_CHANGE_DECLARATION(LogarithmicChange);







	//example definition (LinearChange) - no modification to delta
	//template <typename T> _WEIGHT_CHANGE_DEVICES T LinearChange<T>::adjust(T current, T delta) {
	//	return WeightFunction<T>::learningRate * delta;
	//}


	//example definition, expanding on capabilities
	//template <typename T> _WEIGHT_CHANGE_DEVICES T ClampedLinearChange<T>::adjust(T current, T delta) {
		
	//}


	//shorthand definitions, as no extra detail is needed

	_WEIGHT_CHANGE_FUNCTION(LogisticChange) {
		return (1 / (1 + exp(-delta)));
	};

	_WEIGHT_CHANGE_FUNCTION(LogarithmicChange) {
		return log(delta);
	};






}