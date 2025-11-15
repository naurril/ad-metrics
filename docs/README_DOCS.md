# API Documentation

This directory contains the API reference documentation for AD-Metrics, built using Sphinx.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

Using the build script (recommended):

```bash
./build_docs.sh
```

Using Make directly:

```bash
make html
```

### View Documentation

After building, open `_build/html/index.html` in your browser.

Or run a local server:

```bash
cd _build/html
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

### Live Rebuild (Development)

For automatic rebuilds while editing:

```bash
make livehtml
# Opens http://127.0.0.1:8000
```

### Clean Build

Remove all built documentation:

```bash
make clean
```

## Documentation Structure

```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation index
├── Makefile             # Build automation
├── build_docs.sh        # Build script
├── requirements.txt     # Documentation dependencies
│
├── api/                 # API reference (auto-generated)
│   ├── index.rst
│   ├── detection.rst
│   ├── tracking.rst
│   ├── prediction.rst
│   ├── localization.rst
│   └── ...
│
├── _build/              # Built documentation (generated)
│   └── html/
│
└── *.md                 # Manual documentation (existing)
    ├── METRICS_REFERENCE.md
    ├── DETECTION_METRICS.md
    ├── TRACKING_METRICS.md
    └── ...
```

## Configuration

The documentation is configured in `conf.py` with:

- **Sphinx extensions**: autodoc, napoleon, viewcode, intersphinx
- **Theme**: Read the Docs (sphinx_rtd_theme)
- **Docstring styles**: Google and NumPy formats supported
- **Type hints**: Automatically documented from annotations

## Writing Docstrings

Use Google or NumPy style docstrings in your code:

### Google Style (Recommended)

```python
def calculate_iou_3d(box1, box2):
    """Calculate 3D Intersection over Union between two bounding boxes.
    
    Args:
        box1 (np.ndarray): First bounding box [x, y, z, w, h, l, yaw].
        box2 (np.ndarray): Second bounding box [x, y, z, w, h, l, yaw].
    
    Returns:
        float: IoU score in range [0, 1].
    
    Examples:
        >>> box1 = [0, 0, 0, 4, 2, 1.5, 0]
        >>> box2 = [1, 0, 0, 4, 2, 1.5, 0]
        >>> iou = calculate_iou_3d(box1, box2)
        >>> print(f"IoU: {iou:.4f}")
    """
    pass
```

### NumPy Style

```python
def calculate_iou_3d(box1, box2):
    """Calculate 3D Intersection over Union between two bounding boxes.
    
    Parameters
    ----------
    box1 : np.ndarray
        First bounding box [x, y, z, w, h, l, yaw].
    box2 : np.ndarray
        Second bounding box [x, y, z, w, h, l, yaw].
    
    Returns
    -------
    float
        IoU score in range [0, 1].
    
    Examples
    --------
    >>> box1 = [0, 0, 0, 4, 2, 1.5, 0]
    >>> box2 = [1, 0, 0, 4, 2, 1.5, 0]
    >>> iou = calculate_iou_3d(box1, box2)
    >>> print(f"IoU: {iou:.4f}")
    """
    pass
```

## Troubleshooting

### Import Errors

If you see import errors during build, add the module to `autodoc_mock_imports` in `conf.py`:

```python
autodoc_mock_imports = ['matplotlib', 'open3d', 'your_module']
```

### Module Not Found

Ensure the package is installed or add the path to `sys.path` in `conf.py`:

```python
sys.path.insert(0, os.path.abspath('..'))
```

### Missing Type Hints

Install `sphinx-autodoc-typehints` and enable in `conf.py`:

```python
extensions = ['sphinx_autodoc_typehints']
autodoc_typehints = 'description'
```

## Deployment

### GitHub Pages

1. Build documentation: `./build_docs.sh`
2. Copy `_build/html/*` to your GitHub Pages repository
3. Push to deploy

### Read the Docs

1. Create account at https://readthedocs.org/
2. Import your repository
3. RTD will automatically build on each commit

### Manual Deployment

Build and deploy to any web server:

```bash
./build_docs.sh
rsync -avz _build/html/ user@server:/var/www/docs/
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Read the Docs Theme](https://sphinx-rtd-theme.readthedocs.io/)
- [Napoleon Extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Autodoc Extension](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
