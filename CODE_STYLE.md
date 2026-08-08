# Code Style Reference

This document describes the code style and conventions used in the AI-Aware PCB project for reference and understanding.

## Python Style Guidelines

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following specifics:

### Imports
- Group imports: standard library, third-party, local
- Use absolute imports
- One import per line for clarity

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Module | lowercase with underscores | `compare_boundaries.py` |
| Class | CapWords | `class SafetyModel:` |
| Function | lowercase with underscores | `def classify_region():` |
| Constant | UPPER_CASE | `BETA_MAX = 8.0` |
| Private | leading underscore | `_internal_helper()` |

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def cls_decel(p_t, u_t, w_t, p, lead_faster_safe=True):
    """Classify deceleration scenario safety region.
    
    Determines whether a given gap is safe, unsafe but recoverable, or 
    an unavoidable collision based on perceived gap and vehicle speeds.
    
    Args:
        p_t: Perceived longitudinal gap (m)
        u_t: Ego vehicle speed (m/s)
        w_t: Lead vehicle speed (m/s)
        p: Safety model parameters (P dataclass)
        lead_faster_safe: Short-circuit when lead faster (default: True)
    
    Returns:
        str: One of 'safe', 'unsafe', or 'collision'
    
    Raises:
        ValueError: If speeds are negative or NaN
    """
    if np.isnan(p_t) or np.isnan(u_t) or np.isnan(w_t):
        raise ValueError("Gap and speeds must be valid numbers")
    # ... implementation
```

### Type Hints

Use type hints for function signatures where practical:

```python
def rss_safe_distance(v_r: float, v_f: float, p: 'P') -> float:
    """Calculate RSS safe distance."""
    rho, a = p.RSS_RHO, p.RSS_A_ACCEL
    d = (v_r * rho + 0.5 * a * rho**2 + 
         (v_r + rho * a)**2 / (2 * p.RSS_A_MIN_BRAKE) - 
         v_f**2 / (2 * p.RSS_A_MAX_BRAKE))
    return max(0.0, d)
```

### Line Length

Keep lines under 100 characters. Break long lines logically:

```python
# Good: logical grouping
d = (v_r * rho + 0.5 * a * rho**2 + 
     (v_r + rho * a)**2 / (2 * p.RSS_A_MIN_BRAKE) - 
     v_f**2 / (2 * p.RSS_A_MAX_BRAKE))

# Avoid: arbitrary breaks
d = (v_r * rho + 0.5 * a * rho**2 + (v_r + rho * a)**2 / 
     (2 * p.RSS_A_MIN_BRAKE) - v_f**2 / (2 * p.RSS_A_MAX_BRAKE))
```

### Comments

- Use comments for **why**, not what
- Code should be self-explanatory; comments explain reasoning
- Keep comments concise (typically one line)

```python
# Good: explains the reasoning
# Ego accelerates at ALPHA_MAX during system delay, then ramps brake
dxA = u * p.DELTA_SYS + 0.5 * p.ALPHA_MAX * p.DELTA_SYS**2

# Avoid: redundant with code
# Calculate dxA using DELTA_SYS and ALPHA_MAX
dxA = u * p.DELTA_SYS + 0.5 * p.ALPHA_MAX * p.DELTA_SYS**2
```

### Spacing

- 2 blank lines between module-level definitions
- 1 blank line between methods in a class
- Blank line after imports
- No trailing whitespace

```python
import numpy as np

CONSTANT = 1.0

class MyClass:
    def method1(self):
        pass
    
    def method2(self):
        pass


def module_function():
    pass
```

## File Organization

### Module Structure

```python
"""Module docstring describing purpose and key exports."""

# Standard library imports
import sys
import os

# Third-party imports
import numpy as np
import pandas as pd

# Local imports
from .compare_boundaries import PCB, cls_decel

# Module-level constants
TOLERANCE = 1e-6
TIMEOUT_SECONDS = 300

# ... functions and classes follow
```

### Class Structure

```python
@dataclass
class SafetyModel:
    """Safety envelope model for collision boundary classification.
    
    Attributes:
        name: Model identifier string
        param1: Description (units)
        param2: Description (units)
    """
    name: str
    param1: float
    param2: float = 1.0
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.param1 < 0:
            raise ValueError("param1 must be non-negative")
    
    def compute_boundary(self, speed: float) -> float:
        """Compute safe distance for given speed."""
        return self.param1 * speed**2 / (2 * self.param2)
```

## Testing

All new features must include tests:

```python
import pytest
from Evaluation.compare_boundaries import cls_decel, PCB

def test_cls_decel_safe_region():
    """Verify safe region classification."""
    # High gap should be safe
    result = cls_decel(p_t=50.0, u_t=10.0, w_t=10.0, p=PCB)
    assert result == 'safe'

def test_cls_decel_collision():
    """Verify collision detection."""
    # Very low gap should be collision
    result = cls_decel(p_t=0.5, u_t=15.0, w_t=0.0, p=PCB)
    assert result == 'collision'

def test_cls_decel_invalid_input():
    """Verify input validation."""
    with pytest.raises(ValueError):
        cls_decel(p_t=np.nan, u_t=10.0, w_t=10.0, p=PCB)
```

## Common Patterns

### Iterating Over Scenarios

```python
scenarios = ['deceleration', 'cutin', 'cutout']
for scenario in scenarios:
    results = process_scenario(scenario)
    save_results(results)
```

### Working with DataFrames

```python
df = pd.read_csv('data.csv')

# Use .loc for label-based indexing
safe_rows = df.loc[df['region'] == 'safe']

# Use .iloc for position-based indexing
first_row = df.iloc[0]

# Avoid chaining assignments
df_clean = df.dropna()
df_clean = df_clean[df_clean['speed'] > 0]
```

### Error Handling

```python
def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV with error handling."""
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("CSV file is empty")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
```


### Pre-commit Checks

Before committing, verify:

```bash
# Check PEP 8 compliance
python -m flake8 --max-line-length=100 --exclude=.git,__pycache__ .

# Check imports
python -m isort --check-only .

# Run tests
python -m pytest tests/ -v

# Type checking (optional)
python -m mypy Evaluation/ --ignore-missing-imports
```

## Tools & Configuration

### Recommended Editor Settings

**VS Code** `.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.rulers": [100],
        "editor.formatOnSave": true
    }
}
```

**PyCharm**: Use default PEP 8 settings with line length of 100

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash
python -m flake8 --max-line-length=100 .
if [ $? -ne 0 ]; then
    echo "Flake8 validation failed"
    exit 1
fi
```

## Resources

- [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- [Google Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Real Python: Code Style](https://realpython.com/tutorials/best-practices/)
