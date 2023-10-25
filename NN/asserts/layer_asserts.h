#pragma once
#include "./assert_t.h"
#include "./bulk_assert_t.h"

#include "./type_asserts.h"
#include "./string_asserts.h"


#include "../concepts/layer_trait_concept.h"
#include "../traits/trait_get.h"
namespace NN {
	namespace Asserts {

		


		
		
		/// <summary>
		/// Asserts that there are LayerTraits accessible from the argument
		/// </summary>
		/// <typeparam name="L">Type to assert LayerTraits exist for</typeparam>
		template<typename L>
		struct AssertTraitExists : Assert<Layers::LayerTrait<Traits::getLayerTraits<L>>> {
			static_assert(Layers::LayerTrait<Traits::getLayerTraits<L>>, "Inheritance Error - Expected the layer to contain layer traits - perhaps the interface was not inherited from or not public inheritance?");
		};

		/// <summary>
		/// Runs a series of static assertions on L to ensure that it is a well formed Layer
		/// </summary>
		/// <typeparam name="L">The layer to run assertions on</typeparam>
		template<typename L>
		struct AssertLayer : Assert<> {

			//static assert everything here

			//provide higher-level error messages alongside the lower level ones
			static_assert(struct_assert < AssertIsConstexprString<decltype(L::name)>> , "Fundamental error - LayerType::name should be a FixedString or StringLiteral");
			//static_assert(struct_assert<AssertIsStringCollection<decltype(L::template fullname<L, StringT<'\t'>>)>>, "LayerType::fullname<LayerType,StringT<char 1,...,char n>> should be a StringCollection - check the definition is correct, if shadowing the inherited definition, or that there are no LayerType::name assertion failures");
			static_assert(struct_assert<AssertTraitExists<L>> && struct_assert<BulkAssert<AssertDevice, typename Traits::getLayerTraits<L>::InputDevice, typename Traits::getLayerTraits<L>::OutputDevice>>
				&& struct_assert<BulkAssert<AssertShape, typename Traits::getLayerTraits<L>::InputShape, typename Traits::getLayerTraits<L>::OutputShape>>
				&& struct_assert<BulkAssert<AssertDtype, typename Traits::getLayerTraits<L>::InputDtype, typename Traits::getLayerTraits<L>::OutputDtype>>, "The Layer does not have the correct types for its Inputs and Outputs - see Fundamental errors for specifics");

			//return the result of the static asserts
			static consteval bool stassert() {

				//use strict && because if layer traits did not exist then the rest will fail anyway (except the string assertions but they will likely fail due to a common cause)
				return struct_assert<AssertTraitExists<L>>
					&& struct_assert<BulkAssert<AssertDevice, typename Traits::getLayerTraits<L>::InputDevice, typename Traits::getLayerTraits<L>::OutputDevice>> //assert the trait produces the correct types for devices
					& struct_assert<BulkAssert<AssertShape, typename Traits::getLayerTraits<L>::InputShape, typename Traits::getLayerTraits<L>::OutputShape>> //assert the trait produces the correct types for shapes
					& struct_assert<BulkAssert<AssertDtype, typename Traits::getLayerTraits<L>::InputDtype, typename Traits::getLayerTraits<L>::OutputDtype>> //assert the trait produces the correct types for dtypes
					& struct_assert<AssertIsConstexprString<decltype(L::name)>> //assert the layer has a name of one of the correct types
					;//&& struct_assert<AssertIsStringCollection<decltype(L::template fullname<L, StringT<'\t'>>)>>; //assert the layer has a fullname of the correct type 
			}
		};

	

		//asserts messages for a transition from one layer to another

		//Asserts OutputShape is InputShape
		template<typename OutputShape, typename InputShape>
		struct AssertValidTransitionShape : Assert<struct_assert<AssertEquivalentShapes<OutputShape,InputShape>>> {
			static_assert(struct_assert<AssertEquivalentShapes<OutputShape,InputShape>>, "Sequencing Error - OutputShape is not equivalent to InputShape at a layer transition (see the compiler output for specifics)");
		};

		//Asserts OutputDtype is InputDtype
		template<typename OutputDtype, typename InputDtype>
		struct AssertValidTransitionDtype : Assert<std::same_as<OutputDtype,InputDtype>> {
			static_assert(std::same_as<OutputDtype, InputDtype>, "Sequencing Error - Output datatype does not match input datatype at a layer transition (see the compiler output for specifics)");
		};

		//Asserts OutputDevice is InputDevice
		template<typename PrevOutputDevice, typename NextInputDevice>
		struct AssertValidTransitionDevice : Assert<std::same_as<PrevOutputDevice,NextInputDevice>> {
			static_assert(std::same_as<PrevOutputDevice, NextInputDevice>, "Sequencing Error - Output device does not match input device at a layer transition (see the compiler output for specifics)");
		};

		template<unsigned int OutputThreads, unsigned int InputThreads>
		struct AssertValidTransitionThreads : Assert<OutputThreads==InputThreads> {
			static_assert(OutputThreads == InputThreads, "Sequencing Error - Output threads does not match input threads at a layer transition (see the compiler output for specific)");
		};


		
		
	
		//Asserts that Layer From can transition to Layer To
		template<typename From, typename To>
		struct AssertLegalTransition : Assert<> {
			static consteval bool stassert() {
				bool ret = true;
				//use strict & so that all asserts are run regardless
				ret = ret & AssertLayer<From>::stassert();
				ret = ret & AssertLayer<To>::stassert();


				using FromTraits = Traits::getLayerTraits<From>;
				using ToTraits = Traits::getLayerTraits<To>;

				//assert all are of the correct type
				ret = ret & BulkAssert<AssertShape, typename FromTraits::InputShape, typename FromTraits::OutputShape, typename ToTraits::InputShape, typename ToTraits::OutputShape>::stassert();
				ret = ret & BulkAssert<AssertDevice, typename FromTraits::InputDevice, typename FromTraits::OutputDevice, typename ToTraits::InputDevice, typename ToTraits::OutputDevice>::stassert();
				ret = ret & BulkAssert<AssertDtype, typename FromTraits::InputDtype, typename FromTraits::OutputDtype, typename ToTraits::InputDtype, typename ToTraits::OutputDtype>::stassert();

				//boundary-compatibility asserts

				ret = ret & AssertValidTransitionThreads<FromTraits::nThreads, ToTraits::nThreads>::stassert();
				ret = ret & AssertValidTransitionShape<typename FromTraits::OutputShape, typename ToTraits::InputShape>::stassert();
				ret = ret & AssertValidTransitionDevice<typename FromTraits::OutputDevice, typename ToTraits::InputDevice>::stassert();
				ret = ret & AssertValidTransitionDtype<typename FromTraits::OutputDtype, typename ToTraits::InputDtype>::stassert();

				return ret;
			}
		};
	}
}