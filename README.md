<h1>A strongly typed framework for creating neural networks</h1>
<br>
<p>This is technically my first c++ project, but I did rewrite parts of it many times as I learned more about the language and found better ways of writing things. This does mean that the design of this project was heavily influenced by which areas I was looking to explose and learn more about, resulting in 
the features such as compile-time string concatenation (Which I believe may come in useful later in layer constructions where the inputs are not just input/output tensors), heavy use of template metaprogramming and some curiously recursive template patterns to name a few.</p>

<br>
<p>The main.cpp file contains an example use on the MNIST dataset (which this framework can get 95%+ on currently)</p>
<br>
<h3>File structure</h3>
<ul>
  <li>
    <h4>Asserts</h4>
    <p>Contains objects that force static assertions with a variety of error messages, used as an alternative to concepts when more precise debugging information would be useful</p>
  </li>
  <li>
     <h4>concepts</h4>
    <p>Contains various concepts that are not better placed elsewhere</p>
  </li>
  <li>
    <h4>cuda_helpers</h4>
    <p>Contains various CUDA specific helpers</p>
  </li>
  <li>
    <h4>datasets</h4>
    <p>Contains the dataset interface as well as some specific implementations</p>
  </li>
  <li>
    <h4>dtypes</h4>
    <p>Contains the definitions of various Dtypes - these are wrappers around some numeric-like type that specifies a method of initializing the underlying type as well as some other features</p>
  </li>
  <li>
    <h4>functions</h4>
    <p>Contains various function such as Activation Functions (such as LeakyRelu), Error Measures (for example cross-entropy) and some others</p>
  </li>
  <li>
    <h4>helpers</h4>
    <p>Contains various types/functions that aren't better placed anywhere else</p>
  </li>
  <li>
    <h4>kernels</h4>
    <p>Contains all the processing functions, with specializations for the devices each is able to be run on (Currently CUDA/CPU)</p>
  </li>
  <li>
    <h4>layers</h4>
    <p>Contains the definitions of all the single layers, as well as an interface and some abstract classes</p>
  </li>
  <li>
    <h4>network</h4>
    <p>Contains a class that wraps a layer to act as a network, as well as vaious paramaters for launching the network</p>
  </li>
  <li>
    <h4>references</h4>
    <p>Contains some references that are more strongly typed than T*, as well as limiting mutabilty of the pointer</p>
  </li>
  <li>
    <h4>samples</h4>
    <p>Contains sample network definitions, currently some MNIST networks I have used for testing</p>
  </li>
  <li>
    <h4>sequences</h4>
    <p>Contains layers that wrap an arbitary amount of other layers to provide sequencing to them (or to run them in parallel)</p>
  </li>
  <li>
    <h4>shapes</h4>
    <p>Contains the shape type, along with helper types, which is used to ensure that layers are going to link up correctly and force compiler errors if this is not the case</p>
  </li>
  <li>
    <h4>storage</h4>
    <p>Contains type relevant to the runtime storage of values (inputs/outputs/weights for the most part)</p>
  </li>
  <li>
    <h4>tensors</h4>
    <p>Contains the tensor type, representing a shaped input on a specific device of a specific type, which I probably need to rewrite at some point</p>
  </li>
  <li>
    <h4>threads</h4>
    <p>Contains types relevant to threads and thread-safety</p>
  </li>
  <li>
    <h4>traits</h4>
    <p>Contains traits that allow for example sequences to know the Input/Output devices of the layers they wrap so that they can assert that these are not different between two adjacent layers</p>
  </li>
</ul>
<br>

