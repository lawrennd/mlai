"""
Paramz-lite named parameters for teaching (CIP-000F).

Named parameters with optional positivity transforms, pack/unpack to the
shared 1D optimiser surface, and a Paramz-inspired ``Name | Value | Constraint``
display. Not a Paramz port: no priors, ties, or ``constrain_*`` API zoo.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'POSITIVE',
    'Positive',
    'Param',
    'ParamGroup',
    'format_param_table',
    'format_param_table_html',
]


class Positive:
    """
    Log ↔ exp transform so natural values stay positive.

    Unconstrained coordinate ``u`` maps to natural ``θ = exp(u)``.
    Natural-space gradients scale as ``∂f/∂u = θ · ∂f/∂θ``.
    """

    label = '+ve'

    def to_unconstrained(self, value):
        value = np.asarray(value, dtype=float)
        if np.any(value <= 0.0):
            raise ValueError('Positive transform requires strictly positive values')
        return np.log(value)

    def to_natural(self, unconstrained):
        return np.exp(np.asarray(unconstrained, dtype=float))

    def natural_grad_scale(self, natural_value):
        """Jacobian factor ``d(natural)/d(unconstrained)`` = natural value."""
        return np.asarray(natural_value, dtype=float)


POSITIVE = Positive()


class Param:
    """
    A named parameter holding a natural-scale value.

    :param name: Leaf name (e.g. ``'lengthscale'``)
    :param value: Natural-scale value (scalar or array)
    :param transform: ``None`` or :data:`POSITIVE`
    :param path: Full dotted path for display (e.g. ``'gp.rbf.lengthscale'``)
    """

    def __init__(self, name, value, transform=None, path=None):
        self.name = str(name)
        self.transform = transform
        self.path = path if path is not None else self.name
        self._value = np.asarray(value, dtype=float).copy()
        if self._value.ndim == 0:
            self._value = np.asarray(float(self._value), dtype=float)

    @property
    def value(self):
        """Natural-scale value (scalar float or ndarray)."""
        if self._value.ndim == 0 or self._value.shape == ():
            return float(self._value)
        return self._value.copy()

    @value.setter
    def value(self, new_value):
        arr = np.asarray(new_value, dtype=float)
        if arr.shape != self._value.shape and not (
            self._value.ndim == 0 and arr.ndim == 0
        ):
            # Allow setting scalar into 0-d / size-(,) storage
            if self._value.size == 1 and arr.size == 1:
                arr = np.asarray(float(arr), dtype=float).reshape(self._value.shape)
            else:
                raise ValueError(
                    f"Param {self.path!r}: expected shape {self._value.shape}, "
                    f"got {arr.shape}"
                )
        if self.transform is POSITIVE or isinstance(self.transform, Positive):
            if np.any(arr <= 0.0):
                raise ValueError(
                    f"Param {self.path!r} has +ve constraint; values must be > 0"
                )
        self._value = arr.copy() if arr.ndim else np.asarray(float(arr), dtype=float)

    @property
    def size(self):
        return int(np.asarray(self._value).size)

    @property
    def constraint(self):
        if self.transform is POSITIVE or isinstance(self.transform, Positive):
            return POSITIVE.label
        return ''

    def pack(self):
        """Flatten to unconstrained coordinates for shared optimisers."""
        natural = np.asarray(self._value, dtype=float).ravel()
        if self.transform is None:
            return natural.copy()
        return np.asarray(self.transform.to_unconstrained(natural), dtype=float).ravel()

    def unpack(self, unconstrained):
        """Set natural value from a flat unconstrained vector slice."""
        flat = np.asarray(unconstrained, dtype=float).ravel()
        if flat.size != self.size:
            raise ValueError(
                f"Param {self.path!r}: expected {self.size} values, got {flat.size}"
            )
        if self.transform is None:
            natural = flat
        else:
            natural = self.transform.to_natural(flat)
        self._value = natural.reshape(self._value.shape).copy()
        return self.value

    def scale_natural_gradient(self, natural_grad):
        """Map natural-space gradient to unconstrained space."""
        g = np.asarray(natural_grad, dtype=float).ravel()
        if g.size != self.size:
            raise ValueError(
                f"Param {self.path!r}: expected {self.size} grads, got {g.size}"
            )
        if self.transform is None:
            return g.copy()
        scale = np.asarray(
            self.transform.natural_grad_scale(self._value), dtype=float
        ).ravel()
        return g * scale

    def display_value(self):
        """Compact string for the teaching table Value column."""
        arr = np.asarray(self._value, dtype=float)
        if arr.size == 1:
            return f"{float(arr):.3f}"
        if arr.size <= 4:
            flat = ', '.join(f'{float(v):.3f}' for v in arr.ravel())
            return f"[{flat}]"
        return str(tuple(int(s) for s in arr.shape))


class ParamGroup:
    """
    Ordered collection of :class:`Param` objects with pack/unpack and display.

    :param params: Sequence of :class:`Param`
    :param name: Optional root name (unused in paths if each Param has ``path``)
    """

    def __init__(self, params=None, name=''):
        self.name = name
        self._params = list(params) if params is not None else []

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def add(self, param):
        """Append a :class:`Param`."""
        if not isinstance(param, Param):
            raise TypeError('ParamGroup.add expects a Param')
        self._params.append(param)
        return param

    @property
    def size(self):
        return sum(p.size for p in self._params)

    def pack(self):
        """Concatenate unconstrained coordinates (Netlab-style pak)."""
        if not self._params:
            return np.zeros(0, dtype=float)
        return np.concatenate([p.pack() for p in self._params])

    def unpack(self, unconstrained):
        """
        Write unconstrained vector into natural values (Netlab-style unpak).

        :returns: List of natural values in parameter order
        :rtype: list
        """
        flat = np.asarray(unconstrained, dtype=float).ravel()
        if flat.size != self.size:
            raise ValueError(
                f"Expected {self.size} parameters, got {flat.size}"
            )
        values = []
        idx = 0
        for param in self._params:
            n = param.size
            values.append(param.unpack(flat[idx:idx + n]))
            idx += n
        return values

    def scale_natural_gradients(self, natural_grads):
        """
        Chain-rule map from natural-space grads to unconstrained grads.

        :param natural_grads: 1D array matching :meth:`pack` layout in *natural*
            block sizes (same order/sizes as unconstrained packing)
        """
        flat = np.asarray(natural_grads, dtype=float).ravel()
        if flat.size != self.size:
            raise ValueError(
                f"Expected {self.size} gradients, got {flat.size}"
            )
        out = []
        idx = 0
        for param in self._params:
            n = param.size
            out.append(param.scale_natural_gradient(flat[idx:idx + n]))
            idx += n
        return np.concatenate(out) if out else np.zeros(0, dtype=float)

    def rows(self):
        """Rows for display: ``(path, value_str, constraint)``."""
        return [(p.path, p.display_value(), p.constraint) for p in self._params]

    def __repr__(self):
        return format_param_table(self.rows())

    def _repr_html_(self):
        return format_param_table_html(self.rows())


def format_param_table(rows):
    """
    Text table in Paramz spirit: ``Name | Value | Constraint``.

    :param rows: Iterable of ``(name, value, constraint)``
    """
    rows = list(rows)
    if not rows:
        return 'Name | Value | Constraint\n(no parameters)'
    name_w = max(4, max(len(str(r[0])) for r in rows))
    val_w = max(5, max(len(str(r[1])) for r in rows))
    con_w = max(10, max(len(str(r[2])) for r in rows))
    header = (
        f"{'Name':<{name_w}} | {'Value':>{val_w}} | {'Constraint':<{con_w}}"
    )
    lines = [header, '-' * len(header)]
    for name, value, constraint in rows:
        lines.append(
            f"{str(name):<{name_w}} | {str(value):>{val_w}} | {str(constraint):<{con_w}}"
        )
    return '\n'.join(lines)


def format_param_table_html(rows):
    """HTML table for notebook display (Name / Value / Constraint)."""
    parts = [
        '<table>',
        '<thead><tr><th>Name</th><th>Value</th><th>Constraint</th></tr></thead>',
        '<tbody>',
    ]
    for name, value, constraint in rows:
        parts.append(
            '<tr>'
            f'<td>{_html_escape(name)}</td>'
            f'<td style="text-align:right">{_html_escape(value)}</td>'
            f'<td>{_html_escape(constraint)}</td>'
            '</tr>'
        )
    parts.append('</tbody></table>')
    return ''.join(parts)


def _html_escape(text):
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )
