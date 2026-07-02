---
title: 'Wetting-angle-kit: a Python package for automated wetting contact angle analysis of nanodroplets'
tags:
  - Python
  - Nanodroplets
  - Molecular dynamics
  - Wetting properties
authors:
  - name: Gabriel Taillandier
    orcid: 0009-0006-9544-0982
    affiliation: "1, 2"
  - name: Edoardo Monti
    orcid: 0009-0002-8340-7940
    affiliation: "4"
  - name: Guillaume Brunin
    orcid: 0000-0003-1159-8389
    affiliation: "1"
  - name: Julien Bouquiaux
    orcid: 0000-0003-1982-052X
    affiliation: "1"
  - name: George Froudakis
    orcid: 0000-0002-6907-1822
    affiliation: "2"
  - name: Gian-Marco Rignanese
    orcid: 0000-0002-1422-1205
    affiliation: "1, 3"
  - name: David Waroquiers
    orcid: 0000-0001-8943-9762
    affiliation: "1"


affiliations:
 - name: Matgenix, A6K Advanced Engineering Centre, Charleroi, Belgium.
   index: 1
 - name: Department of Chemistry, University of Crete, Heraklion, Greece
   index: 2
 - name: Institute of Condensed Matter and Nanosciences, UCLouvain, B-1348 Louvain-la-Neuve, Belgium
   index: 3
 - name: Department of Mathematics, Imperial College London, 180 Queen's Gate, London, SW7 2AZ, United Kingdom
   index: 4

date:  May 2026
bibliography: paper.bib
---

# Summary

Wetting-angle-kit is a Python toolkit designed to extract wettability properties,
specifically the contact angle of a droplet on a surface,
from molecular dynamics (MD) simulations of interfaces
between liquids and solid surfaces.

It supports a variety of standard file formats including extended XYZ, LAMMPS,
and ASE-readable trajectories and offers two distinct computational methods
for contact angle analysis.
Furthermore, the package includes robust utilities for statistical post-processing
and data visualization, providing a comprehensive workflow for wettability studies.
This integrated approach reduces the need for custom analysis scripts and improves
reproducibility across different simulation setups.

# Statement of need

Building upon foundational work [@Hautman1997], the methodologies for computing
contact angles from MD simulations have progressed through several key milestones
[@Rafiee2012; @Vega2016; @Carlson2024].
Despite these advancements, the field currently
lacks a standardized, unified tool for comparing and validating the diverse
methods used to derive contact angles. Such fragmentation undermines collaborative
research and reproducibility, as many implementations remain inaccessible or poorly
documented. In addition, the lack of a standardized framework makes it difficult
to benchmark different approaches or assess the impact of methodological choices.

Wetting-angle-kit addresses this critical gap by providing a flexible, open-source package.
It enables the implementation of novel post-processing algorithms
for the extraction and calculation of contact angle, to compare them against accepted techniques,
and to establish a standardized benchmark for MD wettability analysis.

# State of the field

General-purpose MD simulation post-processing tools,
such as OVITO [@Stukowski2010], MDSuite [@Tovey2023],
and MDAnalysis [@Gowers2016; @MichaudAgrawal2011],
provide flexible frameworks for analyzing and visualizing trajectories.
However, they do not include a standardized implementation of
contact angle extraction methods, which are typically developed as custom scripts
tailored to specific systems.

Existing approaches to contact angle computation range from geometric fitting techniques
based on spherical or cylindrical cap approximations [@Hautman1997]
to density-based interface analysis [@Vega2016] and pressure-tensor approaches derived
from planar equilibrium simulations [@Carlson2024], making direct comparison across
methodologies challenging. Wetting-angle-kit complements existing tools by focusing
specifically on wettability analysis and providing a consistent environment for
comparing multiple methods. This design promotes reproducibility and facilitates
the development and/or implementation of other methods.

# Software design

Wetting-angle-kit is organized into three main components:
parsers, analysis, and visualization,
as represented in Figure 1.
This modular organization separates data handling, analysis,
and visualization, allowing components to evolve independently
while simplifying the integration of new features.

![Wetting-angle-kit package structure. The package is composed of three main modules: parsers, analysis, and visualization.](package_overviewDiagram.drawio.pdf){width=90%}

The parser module provides a unified interface for reading trajectory data
from multiple formats, ensuring consistent handling of atomic coordinates,
simulation boxes, and frame information.
This abstraction ensures that analyses are independent of the input format,
enabling consistent workflows across different simulation engines. The parser leverages
established trajectory-reading tools when available, while extended XYZ parsing is
implemented directly within the package. The parser also consistently handles periodic
boundary conditions, ensuring that droplet shapes are correctly reconstructed across
simulation boundaries and avoiding artifacts in interface detection.
This consistency facilitates seamless integration with downstream analysis methods, enabling researchers to easily
incorporate support for additional file formats or simulation engines.

The analysis module provides several composable strategies for extracting contact angles from molecular dynamics trajectories (Fig. 2). Depending on the chosen workflow options, frames can either be analysed individually to preserve temporal information or concatenated into larger batches to improve statistical sampling. This choice allows users to balance temporal resolution against statistical robustness.

Both spherical and cylindrical droplet geometries are supported throughout the analysis workflow [@Scocchi2011]. Spherical droplets provide a direct representation of the three-dimensional cap geometry, whereas cylindrical droplets reduce curvature effects and computational cost through translational symmetry along one direction. The latter geometry is therefore widely used for large systems or when finite-size effects are of primary interest, although it represents an idealized approximation of a fully three-dimensional droplet.

Two approaches are available to estimate the liquid density field. The first uses a grid-based representation, where the local density is obtained by binning atomic positions into spatial bins. The second relies on Gaussian kernel density estimation (KDE), which provides a smooth continuous representation of the density field.

Once the interface or density representation has been constructed, contact angles can be determined using different fitting strategies. Geometric fitting can be applied either to the entire droplet, providing an overall estimate of the contact angle, or independently to multiple slices of the droplet, allowing for the detection of asymmetries and transient shape fluctuations. Alternatively, a coupled-fit approach directly fits a hyperbolic-tangent density model to the density field, simultaneously determining the interface geometry and wall position from a single optimization procedure.

![Schematic representation of the composable strategies in wetting-angle-kit to compute contact angle from a MD trajectory.](schema_methods_analysis.pdf){width=95%}

Additionally, wetting-angle-kit supports two geometric models commonly used
in the literature for droplets: spherical and cylindrical [@Scocchi2011] (see Figure 3).
While the spherical case provides a more direct representation of the droplet curvature,
a cylindrical geometry reduces curvature effects and computational cost by relying on periodic boundary conditions along the cylinder axis,
at the expense of relying on an idealized geometry.

![Geometric representations of droplets used in the analysis: spherical droplet (left) and cylindrical droplet (right).](wetting_angle_kit_sphere_vs_cylinder.pdf){width=90%}

The visualization module includes tools to support interpretation and validation
of analysis results without requiring external post-processing tools.
These tools consists in (1) a contact-angle vs. trajectory timeframe for the slicing analysis,
and (2) a density heatmap based on the binning analysis.

The software architecture relies on abstract base classes to enforce
consistent interfaces and facilitate extensibility.
This design enables users to implement new computational methods while maintaining
compatibility with existing workflows, promoting reuse and method comparison.
It also facilitates the integration of newly developed methods into
an existing and standardized analysis pipeline.

# Research impact statement

Wetting-angle-kit provides a reproducible framework for contact angle analysis
in MD simulations, addressing a common need in studies of nanoscale wetting.
The package has been validated using MD simulations of water droplets on graphene
and polymer substrates, yielding contact angle values consistent
with literature results (e.g., ~93° for graphene, ~110° for PTFE), see Figure 4.
The reported contact angles were obtained by analyzing droplets of increasing sizes
and extrapolating to the macroscopic limit using the modified Young’s equation **ref**,
where the contact angle is related to droplet size, enabling the estimation of the infinite-droplet contact angle through linear extrapolation.
These results are consistent with values reported in the literature, obtained using
similar interatomic potential parameters [@Jorgensen1996] for the MD simulation and SPC/E model for water [@Roberts1999].

![Size-dependent contact angle analysis for water droplets on graphite and PTFE substrates. Values of $\cos(\theta)$ are plotted as a function of the inverse square root of the droplet surface area for droplets containing between 500 and 6000 water molecules. A linear extrapolation following the modified Young’s equation is used to estimate the macroscopic (infinite-droplet) contact angle.](mean_cos_angle_vs_surface_graphite_ptfe.pdf)

By enabling systematic comparison of analysis methods
and providing standardized workflows, wetting-angle-kit supports more robust and
reproducible wettability studies.
Its modular design also facilitates integration into existing simulation pipelines
and encourages community-driven extensions. The package is expected to be particularly
useful for researchers using various types of force fields (classical, ab initio, and machine learned)
or investigating nanoscale interfacial phenomena.

# AI usage disclosure

Generative AI tools (Claude Code with Sonnet 4.6 and Opus 4.7, **XXX Gabriel add yours**) were used in the development of the software,
for drafting and assisting debugging.
Generative AI was used to assist in refining the language,
translation and clarity of the manuscript and docstring.
All AI-assisted contributions were verified and approved by the authors.

# Acknowledgements

This research was supported by the BLESSED project funded by
the European Union under Marie Sklodowska-Curie Actions (Grant Agreement No. 101072578).

Computational resources have been provided by
the Consortium des Équipements de Calcul Intensif (CÉCI),
funded by the FRS-FNRS under Grant No. 2.5020.11 and by the Walloon Region.

# References
