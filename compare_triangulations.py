#!/usr/bin/env python3
import argparse
import numpy as np
import os

from delaunay import sort_triangle
from util import load_triangulation

parser = argparse.ArgumentParser(description="Compare two triangulations")
parser.add_argument("file1", type=str, help="Path to the first triangulation")
parser.add_argument("file2", type=str, help="Path to the second triangulation")


def main():
    args = parser.parse_args()
    for file in [args.file1, args.file2]:
        if not os.path.isfile(file):
            print(f"Error: The file '{file}' does not exist.")
            return

    points1, triangles1 = load_triangulation(args.file1, do_sort=True)
    points2, triangles2 = load_triangulation(args.file2, do_sort=True)
    triangles1 = list(map(sort_triangle, triangles1))
    triangles2 = list(map(sort_triangle, triangles2))

    tr1 = set(triangles1)
    tr2 = set(triangles2)
    identical = tr1 == tr2

    print(f"Points identical: {np.allclose(points1, points2)}")
    print(f"Triangulations identical: {identical}")
    if not identical:
        print(f"trigs1 \\ trigs2: {sorted(list(tr1 - tr2))}")
        print()
        print(f"trigs2 \\ trigs1: {sorted(list(tr2 - tr1))}")


if __name__ == "__main__":
    main()
