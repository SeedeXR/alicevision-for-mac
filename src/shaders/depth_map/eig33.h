#pragma once

// eig33.h — symmetric 3×3 real eigendecomposition helpers.
//
// Extracted from `eig33.metal` (S3) into a header so that other
// MSL translation units can call `eig33_decompose` directly
// (e.g., the PCA plane fit in `depth_sim_map.metal`).
//
// All helpers operate on `thread`-address-space local arrays.
// Precision is FP32 (Apple Silicon GPUs have no native FP64) —
// see the long-form notes in the .metal file's top-of-file
// comment for the precision/numerical-error implications.

#include <metal_stdlib>
using namespace metal;

// FP32 machine epsilon (1 ULP at 1.0 is 2^-23).
constant constexpr float kEig33Eps = 0x1p-23f;

inline float eig33_hypot2(float x, float y)
{
    return sqrt(x * x + y * y);
}

// Householder reduction of a real symmetric 3×3 to tridiagonal
// form. On exit: V is the orthogonal Q with A = Q * T * Q^T;
// d is the diagonal of T; e is the off-diagonal.
inline void eig33_tred2(thread float V[3][3], thread float d[3], thread float e[3])
{
    int i, j, k;
    float scale, h, f, g, hh;

    for (j = 0; j < 3; j++) d[j] = V[2][j];

    for (i = 2; i > 0; i--)
    {
        scale = 0.0f;
        h     = 0.0f;
        for (k = 0; k < i; k++) scale += fabs(d[k]);

        if (scale == 0.0f)
        {
            e[i] = d[i - 1];
            for (j = 0; j < i; j++)
            {
                d[j]    = V[i - 1][j];
                V[i][j] = 0.0f;
                V[j][i] = 0.0f;
            }
        }
        else
        {
            for (k = 0; k < i; k++)
            {
                d[k] /= scale;
                h    += d[k] * d[k];
            }
            f = d[i - 1];
            g = sqrt(h);
            if (f > 0.0f) g = -g;
            e[i]      = scale * g;
            h         = h - f * g;
            d[i - 1]  = f - g;
            for (j = 0; j < i; j++) e[j] = 0.0f;

            for (j = 0; j < i; j++)
            {
                f       = d[j];
                V[j][i] = f;
                g       = e[j] + V[j][j] * f;
                for (k = j + 1; k <= i - 1; k++)
                {
                    g    += V[k][j] * d[k];
                    e[k] += V[k][j] * f;
                }
                e[j] = g;
            }
            f = 0.0f;
            for (j = 0; j < i; j++)
            {
                e[j] /= h;
                f    += e[j] * d[j];
            }
            hh = f / (h + h);
            for (j = 0; j < i; j++) e[j] -= hh * d[j];
            for (j = 0; j < i; j++)
            {
                f = d[j];
                g = e[j];
                for (k = j; k <= i - 1; k++)
                    V[k][j] -= (f * e[k] + g * d[k]);
                d[j]    = V[i - 1][j];
                V[i][j] = 0.0f;
            }
        }
        d[i] = h;
    }

    for (i = 0; i < 2; i++)
    {
        V[2][i] = V[i][i];
        V[i][i] = 1.0f;
        h       = d[i + 1];
        if (h != 0.0f)
        {
            for (k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
            for (j = 0; j <= i; j++)
            {
                g = 0.0f;
                for (k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
                for (k = 0; k <= i; k++) V[k][j] -= g * d[k];
            }
        }
        for (k = 0; k <= i; k++) V[k][i + 1] = 0.0f;
    }
    for (j = 0; j < 3; j++)
    {
        d[j]    = V[2][j];
        V[2][j] = 0.0f;
    }
    V[2][2] = 1.0f;
    e[0]    = 0.0f;
}

// Symmetric tridiagonal QL eigenvalue iteration. Sorts ascending
// on exit. Eigenvectors are columns of V.
inline void eig33_tql2(thread float V[3][3], thread float d[3], thread float e[3])
{
    int i, l, m, iter, k, j;
    float f, g, p, r, dl1, h, c, c2, c3, el1, s, s2;
    float tst1 = 0.0f;

    for (i = 1; i < 3; i++) e[i - 1] = e[i];
    e[2] = 0.0f;

    f = 0.0f;
    for (l = 0; l < 3; l++)
    {
        tst1 = max(tst1, fabs(d[l]) + fabs(e[l]));
        m    = l;
        while (m < 3)
        {
            if (fabs(e[m]) <= kEig33Eps * tst1) break;
            m++;
        }

        if (m > l)
        {
            iter = 0;
            do
            {
                iter = iter + 1;

                g = d[l];
                p = (d[l + 1] - g) / (2.0f * e[l]);
                r = eig33_hypot2(p, 1.0f);
                if (p < 0.0f) r = -r;
                d[l]     = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                dl1      = d[l + 1];
                h        = g - d[l];
                for (i = l + 2; i < 3; i++) d[i] -= h;
                f += h;

                p   = d[m];
                c   = 1.0f;
                c2  = c;
                c3  = c;
                el1 = e[l + 1];
                s   = 0.0f;
                s2  = 0.0f;
                for (i = m - 1; i >= l; i--)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g  = c * e[i];
                    h  = c * p;
                    r  = eig33_hypot2(p, e[i]);
                    e[i + 1] = s * r;
                    s        = e[i] / r;
                    c        = p / r;
                    p        = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);

                    for (k = 0; k < 3; k++)
                    {
                        h           = V[k][i + 1];
                        V[k][i + 1] = s * V[k][i] + c * h;
                        V[k][i]     = c * V[k][i] - s * h;
                    }
                }
                p    = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                if (iter > 64) break;

            } while (fabs(e[l]) > kEig33Eps * tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0.0f;
    }

    for (i = 0; i < 2; i++)
    {
        k = i;
        p = d[i];
        for (j = i + 1; j < 3; j++)
        {
            if (d[j] < p)
            {
                k = j;
                p = d[j];
            }
        }
        if (k != i)
        {
            d[k] = d[i];
            d[i] = p;
            for (j = 0; j < 3; j++)
            {
                p       = V[j][i];
                V[j][i] = V[j][k];
                V[j][k] = p;
            }
        }
    }
}

// Combined entry point. Mirrors the upstream cuda_eigen_decomposition
// signature meaning. On exit, d[] holds the eigenvalues ascending,
// and the columns of V are the corresponding eigenvectors.
inline void eig33_decompose(thread float A[3][3], thread float V[3][3], thread float d[3])
{
    float e[3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            V[i][j] = A[i][j];

    eig33_tred2(V, d, e);
    eig33_tql2(V, d, e);
}
