# Thanks to our ONS colleagues for providing the files for this pipeline

# Step one, Rebuilding from 5 year old dependencies
To understand some of the complexity here we must understand how/why modin df's are being mentioned in the comments found.

Modin is a package to accelerate pandas functionality, but it is not supported with geopandas

