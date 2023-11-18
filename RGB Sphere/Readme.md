# RealTime3DwithPython/RGB Sphere

The RGB Sphere is a morphing vector object for testing three color light sourcing. It starts from a tetrahedron, or, for better results, an icosahedron, the faces of which are equilateral triangles. All the corners (ends of edges) lay on a surface of a single sphere. Then this object can be morphed closer to a full sphere by replacing each triangle with four triangles of equal shape and size, and projecting the new corners (ends of edges), which now are inside the sphere, to the sphere surface. Repeating this a few times results in a 3D vector object looking like the Epcot Center at Disney World. All these objects are also Geodesic Polyhedra.

See also https://oldskoolpython.blogspot.com/

[Note: Originally added 2021. Minor bug fixes due to deprecated NumPy data types in Nov 2023.]
