# TODO: https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
# https://distill.pub/2016/misread-tsne/

# t-SNE (t-distributed stochastic neighbor embedding)
# Visualize high-dimensional data before using it
# How can we possibly reduce the dimensionality of a dataset from an arbitrary number to two or three, which is what we’re doing when we visualize data on a screen?

# The answer lies in the observation that many real-world datasets have a low 
# intrinsic dimensionality, even though they’re embedded in a high-dimensional 
# space. Imagine that you’re shooting a panoramic landscape with your camera, 
# while rotating around yourself. We can consider every picture as a point in 
# a 16,000,000-dimensional space (assuming a 16 megapixels camera). Yet, the 
# set of pictures approximately lie in a three-dimensional space 
# (yaw, pitch, roll). This low-dimensional space is embedded within the high-
# dimensional space in a complex, nonlinear way. Hidden in the data, this 
# structure can only be recovered via specific mathematical methods.

# This is the topic of manifold learning, also called nonlinear dimensionality 
# reduction.