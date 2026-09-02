Trees-and-their-degrees

This is an attempt at understanding a problem that arose in the study of Thompson groups.

There is information about the mathematical context + why we should even care about this in mathcontext.
There are also some files to plot what our objects look like.

We have implemented models for our objects in mathmodels. There are three steps: building a model for the trees, building what we call a "detector map", and computing the degree of the detector map. We suspect that steps two and three do not interact well with each other.

The tests_and_plots folders has various tests to check that every model works. 
It is helpful to plot the objects to have an idea what they look like.
There are separate tests for each of the three main parts of the model.

The scripts folder has some scripts to compute the actual degree of the detector map which is our end goal.
Note: this doesn't actually work that well.
