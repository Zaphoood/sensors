# Sensors

<p align="center">
  <img src="./images/delaunay.png" width="40%"/>
  <img src="./images/connected.png" width="40%"/>
</p>

_This project was created as part of an assignment for the class 'Topological Data Analysis' at the Faculty of Computer
Science of the University of Ljubljana, Slovenia._

## Idea

Given a set of sensors around the globe (e.g. weather stations), represented as points on the unit sphere, we consider
two parameters: Each sensor gathers data in a circle of sensing radius _R_, and can communicate with others which are at
most communication distance _r_ away.

Our goal is to find

- the minimum sensing radius _R_ such that the entire sphere can be observed
- the minimum communication distance _r_ such that the network of sensors is connected.

We compute _R_ by considering the Delaunay triangulation of the points on the sphere, and compute _r_ by
checking all pairwise distances of points.

## Installation

Python >= 3.10 is required. Install the necessary packages with `pip install -r requirements.txt`.

## Usage

Running the program without arguments will generate a sample of points randomly scattered on the sphere.

An input triangulation can be provided via `python main.py --file <file>`, where `<file>` should be a path to an existing triangulation in the following format:

```
<x_1> <y_1> <z_1>
<x_2> <y_2> <z_2>
...
<x_n> <y_n> <z_n>

<a_1> <b_1> <c_1>
<a_2> <b_2> <c_2>
...
<a_m> <b_m> <c_m>
```

Here, `(x_i, y_i, z_i)` are the vertices on the sphere and `(a_i, b_i, c_i)` are the triangles, represented as triples
of integer indices into the list of vertices. The program does not check whether the triangulation is valid nor if all
the points actually lie on the sphere.

You can find an example triangulation in `examples/triangulation.txt`. The expected output for the corresponding
delaunay triangulation is found at `examles/triangulation_delaunay.txt`.

## Creating triangulations

### Plane sweep

To run the plane sweep algorithm on a point cloud, press <kbd>S</kbd>. You may need to remove existing triangles first
by pressing <kbd>X</kbd>. (**Note:** This irrecoverably deletes all current triangles and edges!)

### Manually

To create a triangulation manually, first load a list of vectors from a file.
Then, select three nodes at a time (using <kbd>Shift</kbd>+click) and press <kbd>F</kbd> to fill in the triangle.
Finally, export the triangulation by pressing <kbd>E</kbd>.

## Running edge flip

After creating or loading a triangulation, press <kbd>Space</kbd> to run the edge flip algorithm in order to obtain a
Delaunay triangulation. Press <kbd>E</kbd> in order to export the current triangulation to a file. The file will be
created as `triangulation_<timestamp>.txt`.

## Connecting the sensor network

Press <kbd>C</kbd> to calculate the minimum radius for which the Vietoris-Rips graph (i.e. the 2-skeleton of the
Vietoris-Rips complex) is connected, and show that graph.
