## Trees-and-their-degrees

This is an attempt at understanding a problem that arose in the study of Thompson groups. These groups can be visualized as groups of trees, and we wanted to understand the topology of these spaces of trees. In order to do this, we need to understand the basic building blocks of the trees, aka the paths and junctions, and how all these building blocks are connected together through attaching maps. We focused on trees with simple junctions and trees with double junctions (for more context, see mathcontext). The project in this repository computes the degree of the attaching maps.

### What does this repository contain?

There is information about the mathematical context in mathcontext.

We have implemented models for our objects in the mathmodels package. There are three steps: building a model for the trees, building what we call a "detector map", and computing the degree of the detector map. We have carried out all of these steps in two cases: the case of trees with simple junctions and the case of trees with double junctions.

The examples folder has various plotting functions and their results on some examples. They are separated into junction types and into the three main steps: trees, detector and degree. It is particularly helpful to plot the objects to have an idea what they look like. 

### How to interact with this repository?

The end goal of this project is to compute the degree of a map which is simply a number. It is interesting to see how the steps unfold along the way. Therefore, it is suggested to run the file double_run in examples/double_junction or its counterpart simple_run in examples/simple_junction. Other examples can be found in the examples folder. Usually, the runnable files have example or test in their name. The package mathmodels is managed by [uv](https://docs.astral.sh/uv/getting-started/installation/). The virtual environment is set up by running the two commands `uv lock` and `uv sync`. The files can then be run using the command `uv run python path/file.py`.

### Basic ideas

The goal is to compute the degree of a map between spheres. This is done in the third part of the project using methods from simplicial homology.
The first two parts of the project are dedicated to building the actual map whose degree we want to compute. 
The detector map takes a tree embedded into a cube and detects what kind of junction is at the center of the cube.
There is more information about this in mathcontext/tree_problem. 

### Things to be wary of

Almost all examples are run using N=3. There are two reasons for this: first, it is easier to visualize things in lower dimensions. Secondly, the complexity of the code to compute the degree of a map is exponential in N. Even for N=3, the run-time of the file double_run.py is quite long (several minutes).

