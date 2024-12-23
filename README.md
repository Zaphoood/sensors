# Sensors

## Usage

Run the program as `python main.py <file>`, where `<file>` should be a path to an existing triangulation in the following format:

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
of integer indices into the list of vertices. The program does not check whether the triangulation is valid or that all
the points actually lie on the sphere.

You can find an example triangulation in `examples/triangulation.txt`. The expected output for the corresponding
delaunay triangulation is found at `examles/triangulation_delaunay.txt`.

### Running edge flip

After loading a triangulation, press <kbd>Space</kbd> to run the edge flip algorithm in order to obtain a Delaunay triangulation.
Press <kbd>E</kbd> in order to export the current triangulation to a file. The file will be created as
`triangulation_<timestamp>.txt`.

### Creating triangulations

To create a triangulation, first load a list of vectors from a file.
Then, select three nodes at a time (using <kbd>Shift</kbd>-click) and press <kbd>F</kbd> to fill in the triangle.
Finally, export the triangulation by pressing <kbd>E</kbd>.

Note that currently there is no way to delete faces.
