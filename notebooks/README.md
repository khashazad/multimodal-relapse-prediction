# Using LaTeX with Matplotlib

## Problem

When using the Jupyter notebook to make plots with `matplotlib` and `usetex=True`, you may encounter:
- `RuntimeError: Failed to process string with tex because latex could not be found`
- Missing LaTeX packages errors (e.g., `File 'underscore.sty' not found`)
- Jupyter not finding `pdflatex` even when it's available in your shell (e.g., `which pdflatex` finds it)

The easiest solution is to install LaTeX following the standard installation procedure depending on your system. However, that may not be possible if you don't have sudo access on the machine you are working. The following procedure is one possible workaround.

## Solution: Install a LaTeX Version Locally (example: TinyTeX)

### 1. Install TinyTeX (in your home folder -- no sudo required)
```bash
cd ~
wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh
```

### 2. Add to PATH

Add this line to your `~/.bashrc`:
```bash
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
```

Then reload: `source ~/.bashrc`

### 3. Install Required LaTeX Packages
```bash
tlmgr install underscore
tlmgr install type1cm
tlmgr install cm-super
tlmgr install dvipng
tlmgr install tools
```

### 4. Font Issues

If you encounter font warnings like:
- `findfont: Generic family 'serif' not found because none of the following families were found: Times New Roman`

**Solution**: Check available fonts and use fallbacks:
```python
import matplotlib.font_manager as fm
available_fonts = sorted(set([f.name for f in fm.fontManager.ttflist]))
print(available_fonts)
```

Common font mappings (also see `src/utils.py`):
- `'Times New Roman'` → `'Times'` or `'Liberation Serif'`
- `'Computer Modern'` → `'cmr10'`

Install additional fonts locally by copying `.ttf`/`.ttc` files to `~/.local/share/fonts/` and clearing matplotlib's font cache:
```python
import matplotlib as mpl
import os, glob

cache_dir = mpl.get_cachedir()
for cache in glob.glob(os.path.join(cache_dir, 'fontlist-*.json')):
    os.remove(cache)
```

Once you have installed the fonts that you are trying to use, everything else should work properly.

## Notes

- TinyTeX installs to `~/.TinyTeX/`
- Install additional packages as needed: `tlmgr install <package-name>`
- For missing packages, check matplotlib errors for the package name