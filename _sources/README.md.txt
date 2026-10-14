# MLAI Documentation

This directory contains the source files for the MLAI documentation.

## Building the Documentation

To build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be available in `_build/html/`.

## Viewing the Documentation

- *Local*: Open `_build/html/index.html` in your browser
- *Online*: Visit the GitHub Pages site (configured to serve from the `gh-pages` branch)

## Documentation Structure

- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `tutorials/` - Tutorial documentation
- `api/` - API documentation
- `contributing.rst` - Contributing guidelines
- `tenets.rst` - Project tenets
- `cip/` - Code Improvement Plans
- `backlog/` - Project backlog

## GitHub Pages

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the main branch. The site is served from the `gh-pages` branch. 