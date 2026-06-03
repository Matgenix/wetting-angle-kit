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
    affiliation: "4, 5"
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
 - name: Advanced Technology Division, Toyota Motor Europe NV/SA, Technical Center, Hoge Wei 33B, Zaventem, 1930, Belgium
   index: 5

date:  May 2026
bibliography: paper.bib
---

# Summary

Wetting-angle-kit is a Python toolkit designed to extract wettability properties,
specifically the contact angle of a droplet on a surface,
from molecular dynamics (MD) simulations.
The software is designed for researchers working in MD simulation of interfaces
between liquids and solid surfaces.

It supports a variety of standard file formats including extended XYZ, LAMMPS,
and ASE-readable trajectories and offers two distinct computational methods
for contact angle analysis.
Furthermore, the package includes robust utilities for statistical post-processing
and data visualization, providing a comprehensive workflow for wettability studies.
This integrated approach reduces the need for custom analysis scripts and improves
reproducibility across different simulation setups.

# Statement of need

Building upon foundational work ([@Hautman1997]), the methodologies for computating
contact angles from MD simulations have progressed through several key milestones
[@Rafiee2012; @Vega2016; @Carlson2024].Despite these advancements, the field currently
lacks a standardized, unified tool for comparing and validating the diverse
methods used to derive contact angles. Such fragmentation undermines collaborative
research and reproducibility, as many implementations remain inaccessible or poorly
documented. In addition, the lack of a standardized framework makes it difficult
to benchmark different approaches or assess the impact of methodological choices.

Wetting-angle-kit addresses this critical gap by providing a flexible,
open-source packa It enables the implementation of novel  post-processing algorithms
for the extraction and calculation of contact angle, compare them against accepted techniques,
and establish a standardize benchmark for MD wettability analysis.

# State of the field

General-purpose MD simulation post-processing tools,
such as OVITO [@Stukowski2010], MDSuite [@Tovey2023],
and MDAnalysis [@Gowers2016; @MichaudAgrawal2011],
provide flexible frameworks for analyzing and visualizing trajectories.
However, they do not include a standardized implementation of
contact angle extraction methods, which are typically developed as custom scripts
tailored to specific systems.

Existing approaches to contact angle estimation range from geometric fitting techniques
based on spherical or cylindrical cap approximations [@Hautman1997]
to density-based interface analysis [@Vega2016] and pressure-tensor approaches derived
from planar equilibrium simulations [@Carlson2024], making direct comparison across
methodologies challenging. Wetting-angle-kit complements existing tools by focusing
specifically on wettability analysis and providing a consistent environment for
comparing multiple methods. This design promotes reproducibility and facilitates
the development and/or implementation of other methods.

# Software design

Wetting-angle-kit is organized into three main components:
parsers, contact angle computation methods, and visualization, Fig. \ref{package_overview}.
This modular organization separates data handling, analysis,
and visualization, allowing components to evolve independently
while simplifying the integration of new features.

\begin{figure}[h!]
\centering
\includegraphics[width=0.9\textwidth, trim=100 480 100 200, clip]{package_overviewDiagram.drawio.pdf}
\caption{Wetting-angle-kit, package structure.}
\label{package_overview}
\end{figure}

The parser module provides a unified interface for reading trajectory data
from multiple formats, ensuring consistent handling of atomic coordinates,
simulation boxes, and frame information.
This abstraction ensures that analyses are independent of the input format,
enabling consistent workflows across different simulation engines. The parser leverages
established trajectory-reading tools when available, while extended XYZ parsing is
implemented directly within the package. The parser also consistently handles periodic
boundary conditions, ensuring that droplet shapes are correctly reconstructed across
simulation boundaries and avoiding artifacts in interface detection.

This consistency facilitates seamless integration with downstream analysis methods
and ensures the system's scalability, enabling researchers to easily
incorporate support for additional file formats or simulation engines.

The contact angle computation methods (analysis) module implements
two complementary approaches for contact angle estimation (Fig. \ref{analysis_methods}).

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth, trim=1.5cm 6cm 2.5cm 1cm, clip ]{schema_methods_analysis.pdf}
\caption{Schema of the two contact angle analysis methods.}
\label{analysis_methods}
\end{figure}

The slicing method performs frame-by-frame geometric analysis,
enabling detailed temporal resolution at the cost of higher computational expense.
In practice, this approach provides a local characterization of
the liquid–vapor interface, allowing the detection of asymmetries and transient
deformations of the droplet shape. It is particularly well suited for non-equilibrium
simulations or systems where the droplet deviates from an ideal spherical cap.
In contrast, the binning method constructs time-averaged density fields,
providing a computationally efficient approach suitable for large datasets
and symmetric systems. By averaging particle positions over time,
this method reduces thermal fluctuations and produces a smoother
and more stable interface, making it suitable for extracting
equilibrium contact angles from noisy datasets.
However, this temporal averaging may obscure short-lived fluctuations and
local deviations from ideal geometries.
These two approaches reflect a trade-off between temporal resolution and statistical
robustness, allowing users to select the method best suited to their system.

Additionally, wetting-angle-kit supports two geometric models commonly used
in the literature: spherical and cylindrical [@Scocchi2011] (Fig. \ref{geometries}).
While spherical droplets provide a more direct representation of droplet curvature,
cylindrical geometries reduce curvature effects and computational cost,
at the expense of relying on an idealized geometry.

\begin{figure}[h!]
\centering
\includegraphics[width=0.48\textwidth]{wetting_angle_kit_3d_droplet.pdf}
\hfill
\includegraphics[width=0.48\textwidth]{wetting_angle_kit_cylinder.pdf}
\caption{Geometric representations of droplets used in the analysis: spherical droplet (left) and cylindrical droplet (right).}
\label{geometries}
\end{figure}

The software architecture relies on abstract base classes to enforce
consistent interfaces and facilitate extensibility.
This design enables users to implement new computational methods while maintaining
compatibility with existing workflows, promoting reuse and method comparison.
It also facilitates the integration of newly developed methods into
an existing and standardized analysis pipeline.

Visualization tools are included to support interpretation and validation
of analysis results without requiring external post-processing tools.
These tools provide representations of droplet geometries, enabling users to
directly inspect the quality of interface detection and fitting procedures.
By facilitating visual verification of intermediate and final results,
they help identify potential artifacts or inconsistencies in the analysis and improve
the reliability of extracted contact angles.

# Research impact statement

Wetting-angle-kit provides a reproducible framework for contact angle analysis
in MD simulations, addressing a common need in studies of nanoscale wetting.
The package has been validated using MD simulations of water droplets on graphene
and polymer substrates, yielding contact angle values consistent
with literature results (e.g., ~93° for graphene, ~110° for PTFE), see Fig. \ref{results}.
The reported contact angles were obtained by analyzing droplets of increasing size
and extrapolating to the macroscopic limit using the Modified Young’s relation,
where the contact angle is related to droplet size through a line-tension correction
term, enabling estimation of the infinite-droplet contact angle.
These results are consistent with literature values obtained using
similar carbon-oxygen LJ parameters [@Jorgensen1996].

\begin{figure}[h!]
\centering
\includegraphics[width=0.8\textwidth]{mean_cos_angle_vs_surface_graphite_ptfe.pdf}
\caption{Size-dependent contact angle analysis for water droplets on graphite
 and PTFE substrates. Values of $\cos(\theta)$ are plotted as a function of the inverse square
 root of the droplet surface area for droplets containing between 500 and 6000 water molecules.
 Linear extrapolation following the Modified Young’s relation is used
 to estimate the macroscopic (infinite-droplet) contact angle.}
\label{results}
\end{figure}

By enabling systematic comparison of analysis methods
and providing standardized workflows, the software supports more robust and
reproducible wettability studies.
Its modular design also facilitates integration into existing simulation pipelines
and encourages community-driven extensions. The package is expected to be particularly
useful for researchers using various types of force fields (classical and MLIPs)
or investigating nanoscale interfacial phenomena.

# AI usage disclosure

Generative AI tools were used in the development of the software,
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
