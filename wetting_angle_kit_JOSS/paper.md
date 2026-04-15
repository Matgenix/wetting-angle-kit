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

Wetting-angle-kit is a Python toolkit designed to extract wettability properties, specifically the contact angle of a droplet on a surface, from molecular dynamics (MD) simulations. The software is intended for researchers working in molecular simulation of liquids on solid surfaces.

It supports a variety of standard file formats including extended XYZ, LAMMPS, and ASE-readable trajectories and offers two distinct computational methods for contact angle analysis. Furthermore, the package includes robust utilities for statistical post-processing and data visualization, providing a comprehensive workflow for wettability studies. This integrated approach reduces the need for custom analysis scripts and improves reproducibility across different simulation setups.

# Statement of need

The measurement of contact angles in molecular dynamics simulations has advanced significantly since early methodologies were proposed in 1997, with notable developments occurring in 2012, 2016, and 2024 [@Hautman1997; @Rafiee2012; @Vega2016; @Carlson2024]. Despite these advancements, the field currently lacks a standardized, unified platform for comparing and validating the diverse methods used to derive contact angles. This fragmentation poses challenges to reproducibility and collaborative research. These implementations are often not publicly available or lack sufficient documentation, further limiting reproducibility. In addition, the lack of a standardized framework makes it difficult to benchmark different approaches or assess the impact of methodological choices.

Wetting-angle-kit addresses this gap by providing a flexible, open-source framework. It allows researchers to implement new post-processing methods for contact angle analysis, benchmark them against established techniques, and establish a consistent baseline for wettability analysis in molecular dynamics.

# State of the field

General-purpose molecular simulation post-process analysis tools, such as OVITO [@Stukowski2010], MDSuite [@Tovey2023], and MDAnalysis [@Gowers2016; @MichaudAgrawal2011], provide flexible frameworks for analyzing and visualizing trajectories. However, they do not include a standardized implementation of contact angle extraction methods, which are typically developed as custom scripts tailored to specific systems.

Existing approaches to estimating the contact angle vary significantly, ranging from geometric fitting techniques to density-based methods, and often offer limited interoperability and comparability. Wetting-angle-kit complements existing tools by focusing specifically on wettability analysis and providing a consistent environment for comparing multiple methods. This design promotes reproducibility and facilitates the development and/or implementation of other methods.

# Software design

Wetting-angle-kit is organized into three main components: parsing, analysis, and visualization. This modular design separates data handling, analysis, and visualization, allowing each component to evolve independently and simplifying the integration of new features.

\begin{figure}[h!]
\centering
\includegraphics[width=0.9\textwidth, trim=50 400 50 100, clip]{package_overviewDiagram.drawio.pdf}
\caption{Package workflow scheme}
\end{figure}


The parser module provides a unified interface for reading trajectory data from multiple formats, ensuring consistent handling of atomic coordinates, simulation boxes, and frame information. This abstraction ensures that analyses are independent of the input format, enabling consistent workflows across different simulation engines. The parser also consistently handles periodic boundary conditions, ensuring that droplet shapes are correctly reconstructed across simulation boundaries and avoiding artifacts in interface detection.

This consistency facilitates seamless integration with downstream analysis methods and ensures the system's scalability, enabling researchers to easily incorporate support for additional file formats or simulation engines.

The analysis module implements two complementary approaches for contact angle estimation. The slicing method performs frame-by-frame geometric analysis, enabling detailed temporal resolution at the cost of higher computational expense. In practice, this approach provides a local characterization of the liquid–vapor interface, allowing the detection of asymmetries and transient deformations of the droplet shape. It is particularly well suited for non-equilibrium simulations or systems where the droplet deviates from an ideal spherical cap. In contrast, the binning method constructs time-averaged density fields, providing a computationally efficient approach suitable for large datasets and symmetric systems. By averaging particle positions over time, this method reduces thermal fluctuations and produces a smoother and more stable interface. It is therefore particularly effective for extracting equilibrium contact angles from noisy datasets. However, this temporal averaging may obscure short-lived fluctuations and local deviations from ideal geometries. These two approaches reflect a trade-off between temporal resolution and statistical robustness, allowing users to select the method best suited to their system. All methods must support the two main geometric models: **spherical** (for spherical cap droplets) and **cylindrical** (for filament-like droplets, analyzed along a specific axis) [@Scocchi2011].

\begin{figure}[h!]
\centering
\includegraphics[width=0.48\textwidth]{wetting_angle_kit_3d_droplet.pdf}
\hfill
\includegraphics[width=0.48\textwidth]{wetting_angle_kit_cylinder.pdf}
\caption{Geometric representations of droplets used in the analysis: spherical droplet (left) and cylindrical droplet (right).}
\end{figure}

The software architecture relies on abstract base classes to enforce consistent interfaces and facilitate extensibility. This design enables users to implement new analysis methods while maintaining compatibility with existing workflows, promoting reuse and method comparison. It also facilitates the integration of newly developed methods into an existing and standardized analysis pipeline.

Visualization tools are included to support interpretation and validation of analysis results without requiring external post-processing tools.


# Theoretical framework: Modified Young’s Equation

To extract the macroscopic contact angle from nanoscale measurements, the relationship between the measured contact angle ($\theta$) and the droplet size is analyzed using the Modified Young’s Equation. This relationship accounts for line tension effects, which are significant at the nanoscale. This formulation is commonly used to account for finite-size effects in nanoscale droplets. The equation is linearized to facilitate extrapolation:

$$\cos\theta = \cos\theta_\infty - \frac{\tau}{\gamma_{LV}} \cdot \frac{1}{r_B}$$



By plotting $\cos\theta$ against the inverse of the contact radius (or an equivalent geometric parameter derived from contact area $A$), the data yields a linear trend. The slope of this line corresponds to the influence of line tension ($\tau$) and surface tension ($\gamma_{LV}$), while the intercept provides the contact angle of an infinite droplet ($\cos\theta_\infty$). This regression allows for the extrapolation of fundamental wettability properties from finite-sized nanodroplets.

# Research impact statement

Wetting-angle-kit provides a reproducible framework for contact angle analysis in molecular simulations, addressing a common need in studies of nanoscale wetting. The package has been validated using MD simulations of water droplets on graphene and polymer substrates, yielding contact angle values consistent with literature results (e.g., ~93° for graphene, ~110° for PTFE). This result is consistent with literature values obtained using similar carbon-oxygen LJ parameters [@Jorgensen1996].

\begin{figure}[h!]
\centering
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{mean_cos_angle_vs_surface_graphite.pdf}
    \caption{Graphite}
\end{minipage}
\hfill
\begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\linewidth]{mean_cos_angle_vs_surface_ptfe.pdf}
    \caption{PTFE}
\end{minipage}
\end{figure}
By enabling systematic comparison of analysis methods and providing standardized workflows, the software supports more robust and reproducible wettability studies. Its modular design also facilitates integration into existing simulation pipelines and encourages community-driven extensions. The package is expected to be particularly useful for researchers using various types of force fields (classical and MLIPs) or investigating nanoscale interfacial phenomena.


# AI usage  disclosure
Generative AI tools were used in the development of the software, for drafting and assist debugging. Generative AI was used to assist in refining the language, traduction and clarity of the manuscript.

# Acknowledgements

MSCA fellowship ..



Computational resources have been provided by the Consortium des Équipements de Calcul Intensif (CÉCI), funded by the FRS-FNRS under Grant No. 2.5020.11 and by the Walloon Region.



# References
