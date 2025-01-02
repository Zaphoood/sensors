# Sensors

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
