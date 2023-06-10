# Hodge-Laplacian renormalization group 
Python code to perform Hodge topological renormalization of scale-free simplicial complexes.

The scale-free simplicial complexes are generated using the "Network Geometry with Flavor" outlined in G. Bianconi and C. Rahmede, "Network geometry with flavor: From complexity to quantum geometry", Phys. Rev. E 93, 032315, 2016

## List of files
- ``stat_scan.py``: generates multiple NGF scale-free simplicial complexes, renormalizes them with all the Hodge Laplacians for different times and saves the degree distributions;
- ``stat_analysis.py``: plots the results of ``stat_scan.py``.

## List of functions
- ``NGF``: generates an NGF simplicial complex of dimension 1,2,3 or 4;
- ``boundary_matrices_3``: returns the first four boundary matrices in a sparse format;
- ``generalized_degree``: computes the generalized degrees of a simplicial complex (ex. for dim = 2, faces in each node and faces in each edge);
- ``plot_complex`` : plots nodes, edges and faces of a given simplicial complex;
- ``compute_heat``: returns the values of the specific heat of a given Hodge Laplacian on a time range;
- ``renormalize_simplicial_VARIANTS``: performs a single step of the Hodge-Laplacian renormalization flow on a given simplicial complex.
