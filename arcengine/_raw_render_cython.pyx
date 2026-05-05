# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Cython port of Camera._raw_render's inner loop.

This module is built out-of-tree with `cythonize -i arcengine/_raw_render_cython.pyx`
and produces a compiled .so / .pyd alongside the source. arcengine/camera.py
tries to import it and falls back to a pure-Python implementation if the
extension isn't available.

Why a Cython port: profiling 4f9f05b showed Camera._raw_render was ~60% of
cumulative time on agent-eval workloads (raw=True, RenderMode.ALL), dominated
by per-sprite Python max/min calls and masked-write fancy-indexing. NumPy-batch
attempts regressed at this N (see handoff 2026-05-05-002) because per-numpy-op
overhead exceeds per-Python-builtin overhead at N≈14-28. Cython removes the
per-iteration overhead at the source.
"""
import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def blit_sprites(
    cnp.int8_t[:, ::1] output,
    list sorted_sprites,           # list of Sprite objects in layer order
    int cam_x,
    int cam_y,
    int view_w,
    int view_h,
):
    """Blit a list of sprites into `output` with per-pixel transparency.

    Takes the sprite list directly (rather than pre-extracted xs/ys arrays
    plus rendered-pixel list) so the caller doesn't pay the np.fromiter cost
    twice per frame. Sprite attribute access from Cython is still
    Python-level, but it's the same number of attr lookups as before — what
    we save is the int32 array allocation and per-element fromiter dispatch.

    Negative pixel values are treated as transparent.
    """
    cdef Py_ssize_t n = len(sorted_sprites)
    cdef Py_ssize_t i
    cdef int rel_x, rel_y, sprite_w, sprite_h
    cdef int dx0, dx1, dy0, dy1
    cdef int sx0, sy0
    cdef int dy, dx, sy
    cdef int sprite_x, sprite_y
    cdef cnp.int8_t pixel
    # `const` lets the memoryview bind to read-only arrays (Sprite.render()
    # marks its cache read-only to enforce no-mutation contract).
    cdef const cnp.int8_t[:, ::1] sp

    for i in range(n):
        sprite = sorted_sprites[i]
        sp = sprite.render()
        sprite_h = sp.shape[0]
        sprite_w = sp.shape[1]

        sprite_x = sprite._x
        sprite_y = sprite._y
        rel_x = sprite_x - cam_x
        rel_y = sprite_y - cam_y

        # Destination range (in output) clipped to viewport
        dx0 = rel_x if rel_x > 0 else 0
        dy0 = rel_y if rel_y > 0 else 0
        dx1 = rel_x + sprite_w
        if dx1 > view_w:
            dx1 = view_w
        dy1 = rel_y + sprite_h
        if dy1 > view_h:
            dy1 = view_h

        if dx1 <= dx0 or dy1 <= dy0:
            continue

        # Source offset within sprite (when sprite extends past viewport edge)
        sx0 = -rel_x if rel_x < 0 else 0
        sy0 = -rel_y if rel_y < 0 else 0

        # Masked blit: copy non-transparent pixels.
        for dy in range(dy0, dy1):
            sy = sy0 + (dy - dy0)
            for dx in range(dx0, dx1):
                pixel = sp[sy, sx0 + (dx - dx0)]
                if pixel >= 0:
                    output[dy, dx] = pixel
