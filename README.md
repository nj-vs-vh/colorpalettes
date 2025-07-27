# colorpalettes

Tools to generate custom colormaps and colorpalettes.

## WIP

- color metric
- colorblind filters
- total loss function:
  - color distinctness
  - colorblinded color distinctness (most common types and filters from distinctpy)
  - lightness/value soft band around 0.8 (tab10 and check others)
  - how to treat saturation? keep free? very soft constraints to stimulate pretty colors?
  - perceptual uniformity?
  - tight first color pegging
  - peg centroid to white?
  - restrict to RGB gamut
- seeding:
  - \# of total colors
  - \# of "commonly used" colors
  - start with opposite/triangle/square/other shape in hue space
  - first color