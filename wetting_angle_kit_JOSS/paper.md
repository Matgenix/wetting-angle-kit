---
title: 'Wetting-angle-kit: a Python package to streamline the computation of wetting angles of nanoparticles in liquids'
tags:
  - Python
  - Nanodroplets
  - Molecular dynamics
  - Wetting properties
authors:
  - name: Gabriel Taillandier
    orcid: 0009-0006-9544-0982
    affiliation: "1, 2"
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:
  - name:
    orcid:
    affiliation:

affiliations:
 - name: Matgenix, A6K Advanced Engineering Centre, Charleroi, Belgium.
   index: 1
 - name: Department of Chemistry, University of Crete, Heraklion, Greece
   index: 2
 - name: Institute of Condensed Matter and Nanosciences, Université catholique de Louvain, B-1348 Louvain-la-Neuve, Belgium
   index: 3
 - name: Imperial
   index: 4
 - name: Toyota
   index: 5

date: January 2026
bibliography: paper.bib
---

# Summary

Wetting-angle-kit is a Python toolkit designed to extract wettability properties, specifically the contact angle of a droplet on a surface, from molecular dynamics (MD) simulations. The software is intended for researchers working in molecular simulation of liquids on top of solid surfaces.

It supports a variety of standard file formats including extended XYZ, LAMMPS, and ASE-readable trajectories and offers two distinct computational methods for contact angle analysis. Furthermore, the package includes robust utilities for statistical post-processing and data visualization, providing a comprehensive workflow for wettability studies.

# Statement of need

The measurement of contact angles in molecular dynamics simulations has advanced significantly since early methodologies were proposed in 1997, with notable developments occurring in 2012, 2016, and 2024 [1-4]. Despite these advancements, the field currently lacks a standardized, unified platform for comparing and validating the diverse methods used to derive contact angles. This fragmentation poses challenges to reproducibility and collaborative research.

Wetting-angle-kit addresses this gap by providing a flexible, open-source framework. It allows researchers to implement new post-processing of the MD simulation of contact angle, benchmark them against established techniques, and establish a consistent baseline for wettability analysis in molecular dynamics.

# State of the field

General-purpose molecular simulation post-process analysis tools, such as MDAnalysis and MDsuite, provide flexible frameworks for analyzing and visualizing trajectories. However, they do not include a standardized implementation of contact angle extraction methods, which are typically developed as custom scripts tailored to specific systems.

Existing approaches to estimating the contact angle vary significantly, ranging from geometric fitting techniques to density-based methods, and often offer limited interoperability and comparability. Wetting-angle-kit complements existing tools by focusing specifically on wettability analysis and providing a consistent environment for comparing multiple methods. This design promotes reproducibility and facilitates the development and/or implementation of other methods.

# Software design

Wetting-angle-kit is organized into three main components: parsing, analysis, and visualization. This modular design separates data handling, analysis, and visualization, allowing each component to evolve independently and simplifying the integration of new features.

![3D spherical droplet scheme](package_overviewDiagram.drawio.pdf){width=50%}


The parser module provides a unified interface for reading trajectory data from multiple formats, ensuring consistent handling of atomic coordinates, simulation boxes, and frame information. This abstraction ensures that analyses are independent of the input format, enabling consistent workflows across different simulation engines.

This consistency facilitates seamless integration with downstream analysis methods and ensures the system's scalability, enabling researchers to easily incorporate support for additional file formats or simulation engines.

The analysis module implements two complementary approaches for contact angle estimation. The slicing method performs frame-by-frame geometric analysis, enabling detailed temporal resolution at the cost of higher computational expense. In contrast, the binning method constructs time-averaged density fields, providing a computationally efficient approach suitable for large datasets and symmetric systems. Supporting both methods allows users to balance accuracy and performance depending on their application. All methods must support the two main geometric models: **spherical** (for spherical cap droplets) and **cylindrical** (for filament-like droplets, analyzed along a specific axis) [@Scocchi2011].

![3D spherical droplet scheme](wetting_angle_kit_3d_droplet.pdf){width=50%}

![Sliced droplet scheme](wetting_angle_kit.pdf){width=50%}

The software architecture relies on abstract base classes to enforce consistent interfaces and facilitate extensibility. This design enables users to implement new analysis methods while maintaining compatibility with existing workflows, promoting reuse and method comparison.

Visualization tools are included to support interpretation and validation of analysis results without requiring external post-processing tools.


# Theoretical framework: Modified Young’s Equation

To extract the macroscopic contact angle from nanoscale measurements, the relationship between the measured contact angle ($\theta$) and the droplet size is analyzed using the Modified Young’s Equation. This relationship accounts for line tension effects, which are significant at the nanoscale. The equation is linearized to facilitate extrapolation:

$$\cos\theta = \cos\theta_\infty - \frac{\tau}{\gamma_{LV}} \cdot \frac{1}{r_B}$$



By plotting $\cos\theta$ against the inverse of the contact radius (or an equivalent geometric parameter derived from contact area $A$), the data yields a linear trend. The slope of this line corresponds to the influence of line tension ($\tau$) and surface tension ($\gamma_{LV}$), while the intercept provides the contact angle of an infinite droplet ($\cos\theta_\infty$). This regression allows for the extrapolation of fundamental wettability properties from finite-sized nanodroplets.

# Research impact statement

Wetting-angle-kit provides a reproducible framework for contact angle analysis in molecular simulations, addressing a common need in studies of nanoscale wetting. The package has been validated using MD simulations of water droplets on graphene and polymer substrates, yielding contact angle values consistent with literature results (e.g., ~93° for graphene, ~110° for PTFE). This result is consistent with literature values obtained using similar carbon-oxygen LJ parameters [@Jorgensen1996].

![Mean cos angle vs surface for graphite](menscosnalge_vs_surface_graphite.pdf){width=50%}

![Mean cos angle vs surface for PTFE](menscosnalge_vs_surface_ptfe.pdf){width=50%}

By enabling systematic comparison of analysis methods and providing standardized workflows, the software supports more robust and reproducible wettability studies. Its modular design also facilitates integration into existing simulation pipelines and encourages community-driven extensions. The package is expected to be particularly useful for researchers using various types of force fields (classical and MLIPs) or investigating nanoscale interfacial phenomena.


# AI usage  disclosure
Generative AI tools were used in the development of the software, for drafting and assist debugging. Generative AI was used to assist in refining the language, traduction and clarity of the manuscript.

# Acknowledgements

MSCA fellowship ..



Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the FRS-FNRS under Grant No. 2.5020.11 and by the Walloon Region.



# References
