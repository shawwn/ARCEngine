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
    list rendered_pixels,         # list of 2D int8 ndarrays, one per sprite (already in layer order)
    cnp.int32_t[::1] xs,          # sprite world x positions
    cnp.int32_t[::1] ys,          # sprite world y positions
    int cam_x,
    int cam_y,
    int view_w,
    int view_h,
):
    """Blit a list of sprites into `output` with per-pixel transparency.

    Equivalent to the per-sprite loop in Camera._raw_render but with the
    bbox/clip math and the masked write running in C-speed Cython instead of
    Python. Negative pixel values are treated as transparent.

    Arguments:
        output: 2D int8 framebuffer of shape (view_h, view_w), pre-filled with
                background color. Mutated in place.
        rendered_pixels: list of 2D int8 ndarrays (sprite.render() outputs).
                Order matters: lower-layer sprites first, painted over by later.
        xs, ys: sprite world positions.
        cam_x, cam_y: camera origin in world coordinates.
        view_w, view_h: framebuffer size.
    """
    cdef Py_ssize_t n = xs.shape[0]
    cdef Py_ssize_t i
    cdef int rel_x, rel_y, sprite_w, sprite_h
    cdef int dx0, dx1, dy0, dy1
    cdef int sx0, sy0
    cdef int dy, dx, sy
    cdef cnp.int8_t pixel
    # `const` lets the memoryview bind to read-only arrays (Sprite.render()
    # marks its cache read-only to enforce no-mutation contract).
    cdef const cnp.int8_t[:, ::1] sp

    for i in range(n):
        sp = rendered_pixels[i]
        sprite_h = sp.shape[0]
        sprite_w = sp.shape[1]

        rel_x = xs[i] - cam_x
        rel_y = ys[i] - cam_y

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
