#pragma once
#include "./wrapper/network_wrapper.h"
#include "../traits/trait_get.h"
#include "../sequences/sequence.h"
#include "../layers/device_transformations/device_change.h"
#include "../functions/error_measures/halved_squared_error.h"
#include "../traits/network_traits.h"
namespace NN {

	/// <summary>
	/// A wrapper around one or more layers which provides network functions
	/// </summary>
	/// <typeparam name="LayerT">The layer to wrap</typeparam>
	/// <typeparam name="ErrorMeasure">The measure of error in the network's predictions</typeparam>
	template<typename LayerT, typename ErrorMeasure = ErrorMeasures::HalvedSquaredError>
	class Network;

	//Both Input and Output are CPU
	template<typename LayerT,  typename ErrorMeasure> requires (Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::InputDevice> && Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::OutputDevice>)
	class Network<LayerT, ErrorMeasure> : public Networks::NetworkWrapper<LayerT, ErrorMeasure>, public Traits::NetworkTraits<LayerT> {
	public:

		using Network::NetworkWrapper<LayerT, ErrorMeasure>::NetworkWrapper;
	};

	//Input is CPU, but Output is not - therefore wrap a sequence containing a DeviceChange to CPU at the end
	template<typename LayerT, typename ErrorMeasure> requires (Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::InputDevice> && !Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::OutputDevice>)
	class Network <LayerT, ErrorMeasure> : public Networks::NetworkWrapper<Sequence<
		LayerT,
		Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::OutputDtype,
			typename Traits::getLayerTraits<LayerT>::OutputShape,
			typename Traits::getLayerTraits<LayerT>::OutputDevice,
			Devices::CPU,
			Traits::getLayerTraits<LayerT>::nThreads
		>
	>,ErrorMeasure>, //inherit from the networkwrapper of a sequence that ends on CPU
	public Traits::NetworkTraits< Sequence<
		LayerT,
		Layers::DeviceChange<
		typename Traits::getLayerTraits<LayerT>::OutputDtype,
		typename Traits::getLayerTraits<LayerT>::OutputShape,
		typename Traits::getLayerTraits<LayerT>::OutputDevice,
		Devices::CPU,
		Traits::getLayerTraits<LayerT>::nThreads
		>
		>> //inherit network traits

	{
	public:
		static constexpr const StringLiteral name = StringLiteral("Network (adjusted to end on CPU)");

		using Networks::NetworkWrapper<Sequence<
			LayerT,
			Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::OutputDtype,
			typename Traits::getLayerTraits<LayerT>::OutputShape,
			typename Traits::getLayerTraits<LayerT>::OutputDevice,
			Devices::CPU,
			Traits::getLayerTraits<LayerT>::nThreads
			>
			>,ErrorMeasure>::NetworkWrapper;
	
	};

	//Input is not CPU, but Output is - therefore wrap a sequence containing a DeviceChange from CPU at the start
	template<typename LayerT,  typename ErrorMeasure> requires (!Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::InputDevice>&& Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::OutputDevice>)
	class Network<LayerT, ErrorMeasure> : public Networks::NetworkWrapper<Sequence<
		Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::InputDtype,
			typename Traits::getLayerTraits<LayerT>::InputShape,
			Devices::CPU,
			typename Traits::getLayerTraits<LayerT>::InputDevice,
			Traits::getLayerTraits<LayerT>::nThreads
		>,
		LayerT
	>,ErrorMeasure>, public Traits::NetworkTraits< Sequence<
		Layers::DeviceChange<
		typename Traits::getLayerTraits<LayerT>::InputDtype,
		typename Traits::getLayerTraits<LayerT>::InputShape,
		Devices::CPU,
		typename Traits::getLayerTraits<LayerT>::InputDevice,
		Traits::getLayerTraits<LayerT>::nThreads
		>,
		LayerT
	>> {
	public:

		static constexpr const StringLiteral name = StringLiteral("Network (adjusted to start on CPU)");

		using Networks::NetworkWrapper<Sequence<
			Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::InputDtype,
			typename Traits::getLayerTraits<LayerT>::InputShape,
			Devices::CPU,
			typename Traits::getLayerTraits<LayerT>::InputDevice,
			Traits::getLayerTraits<LayerT>::nThreads
			>,
			LayerT
			>,ErrorMeasure>::NetworkWrapper;
	};

	//Both Input and Output are not CPU - therefore wrap a sequence containing a DeviceChange from CPU at the start, and to CPU at the end
	template<typename LayerT, typename ErrorMeasure> requires (!Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::InputDevice> && !Devices::CPUDevice<typename Traits::getLayerTraits<LayerT>::OutputDevice>)
	class Network<LayerT, ErrorMeasure> : public Networks::NetworkWrapper<Sequence<
		Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::InputDtype,
			typename Traits::getLayerTraits<LayerT>::InputShape,
			Devices::CPU,
			typename Traits::getLayerTraits<LayerT>::InputDevice,
			Traits::getLayerTraits<LayerT>::nThreads
		>,
		LayerT,
		Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::OutputDtype,
			typename Traits::getLayerTraits<LayerT>::OutputShape,
			typename Traits::getLayerTraits<LayerT>::OutputDevice,
			Devices::CPU,
			Traits::getLayerTraits<LayerT>::nThreads
		>
	>,ErrorMeasure>, public Traits::NetworkTraits< Sequence<
		Layers::DeviceChange<
		typename Traits::getLayerTraits<LayerT>::InputDtype,
		typename Traits::getLayerTraits<LayerT>::InputShape,
		Devices::CPU,
		typename Traits::getLayerTraits<LayerT>::InputDevice,
		Traits::getLayerTraits<LayerT>::nThreads
		>,
		LayerT,
		Layers::DeviceChange<
		typename Traits::getLayerTraits<LayerT>::OutputDtype,
		typename Traits::getLayerTraits<LayerT>::OutputShape,
		typename Traits::getLayerTraits<LayerT>::OutputDevice,
		Devices::CPU,
		Traits::getLayerTraits<LayerT>::nThreads
		>
	>> {
	public:
		static constexpr const StringLiteral name = StringLiteral("Network (adjusted to start and end on CPU)");

		using Networks::NetworkWrapper<Sequence<
			Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::InputDtype,
			typename Traits::getLayerTraits<LayerT>::InputShape,
			Devices::CPU,
			typename Traits::getLayerTraits<LayerT>::InputDevice,
			Traits::getLayerTraits<LayerT>::nThreads
			>,
			LayerT,
			Layers::DeviceChange<
			typename Traits::getLayerTraits<LayerT>::OutputDtype,
			typename Traits::getLayerTraits<LayerT>::OutputShape,
			typename Traits::getLayerTraits<LayerT>::OutputDevice,
			Devices::CPU,
			Traits::getLayerTraits<LayerT>::nThreads
			>
			>,ErrorMeasure>::NetworkWrapper;
	};

}