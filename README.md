Trees-and-their-degrees

This is an attempt at understanding a problem that arose in the study of Thompson groups.

What does this repository contain?

There is information about the mathematical context in mathcontext.

We have implemented models for our objects in mathmodels. There are three steps: building a model for the trees, building what we call a "detector map", and computing the degree of the detector map. We have carried out all of these steps in two cases: the case of trees with simple junctions and the case of trees with double junctions.

The examples folder has various plotting functions and their results on some examples. It is particularly helpful to plot the objects to have an idea what they look like. There are examples for every stage: we can plot the trees in step 1, plot the detector (in various ways) in step 2, and compute the degree in step 3.

How to interact with this repository?

The end goal of this project is to compute a degree which is simply a number. It is interesting to see how the steps unfold along the way, though.
Therefore, it is suggested to run the file example_run_a in examples_double or its counterpart.

Basic ideas

There is more information about this in mathcontext/tree_problem, however here is a quick breakdown of what is done. The goal is to compute the degree of a map between spheres. This is done in the third part of the project using methods from simplicial homology.

The first two parts of the project are dedicated to building the actual map whose degree we want to compute. 
