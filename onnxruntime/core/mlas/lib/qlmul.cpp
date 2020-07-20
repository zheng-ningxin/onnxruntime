/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qlmul.cpp

Abstract:

    This module implements routines to quantize linear mul.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

#if defined(MLAS_NEON_INTRINSICS)

#elif defined(MLAS_SSE2_INTRINSICS)

template<typename DataType, bool IsScalarB>
static
void
MlasQLinearMulKernel(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N
    )
{
    const auto VectorZeroPointA = MlasBroadcastInt32x4(ZeroPointA);
    const auto VectorZeroPointB = MlasBroadcastInt32x4(ZeroPointB);
    const auto VectorZeroPointC = MlasBroadcastInt32x4(ZeroPointC);
    const auto VectorScaleRatio = MlasBroadcastFloat32x4(ScaleA * ScaleB / ScaleC);

    MLAS_INT32X4 va_lo, va_hi, vb_lo, vb_hi;
    if (IsScalarB) {
        vb_lo = _mm_sub_epi32(_mm_set1_ps((int32_t)*InputB), VectorZeroPointB);
        vb_hi = vb_lo;
    }

    while (N >= 8) {
        const auto va_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)InputA);
        InputA += 8;
        const auto va_i16x8 = _mm_unpacklo_epi8(va_low_half, va_low_half);
        va_lo = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(va_i16x8, va_i16x8), 24), VectorZeroPointA);
        va_hi = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(va_i16x8, va_i16x8), 24), VectorZeroPointA);

        if (!IsScalarB) {
            const auto vb_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)InputB);
            InputB += 8;
            const auto vb_i16x8 = _mm_unpacklo_epi8(vb_low_half, vb_low_half);
            vb_lo = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(vb_i16x8, vb_i16x8), 24), VectorZeroPointB);
            vb_hi = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(vb_i16x8, vb_i16x8), 24), VectorZeroPointB);
        }

        auto r_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_mul_epi32(va_lo, vb_lo)), VectorScaleRatio)), VectorZeroPointC);
        auto r_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_mul_epi32(va_hi, vb_hi)), VectorScaleRatio)), VectorZeroPointC);

        const auto vc_i16x8 = _mm_packs_epi32(r_lo, r_hi);
        MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

        N -= 8;
        _mm_storel_epi64((MLAS_INT32X4*)OutputC, vc);
        OutputC += 8;
    }

    if (N > 0) {
        uint8_t TailData[8] = { 0 };

        MlasCopyTailBytes(TailData, (const uint8_t*)InputA, N);
        const auto va_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)TailData);
        const auto va_i16x8 = _mm_unpacklo_epi8(va_low_half, va_low_half);
        va_lo = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(va_i16x8, va_i16x8), 24), VectorZeroPointA);
        va_hi = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(va_i16x8, va_i16x8), 24), VectorZeroPointA);

        if (!IsScalarB) {
            MlasCopyTailBytes(TailData, (const uint8_t*)InputB, N);
            const auto vb_low_half = _mm_loadl_epi64((const MLAS_INT32X4*)TailData);
            const auto vb_i16x8 = _mm_unpacklo_epi8(vb_low_half, vb_low_half);
            vb_lo = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpacklo_epi16(vb_i16x8, vb_i16x8), 24), VectorZeroPointB);
            vb_hi = _mm_sub_epi32(MlasShiftRightInt32<DataType>(_mm_unpackhi_epi16(vb_i16x8, vb_i16x8), 24), VectorZeroPointB);
        }

        auto r_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_mul_epi32(va_lo, vb_lo)), VectorScaleRatio)), VectorZeroPointC);
        auto r_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_mul_epi32(va_hi, vb_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_i16x8 = _mm_packs_epi32(r_lo, r_hi);
        MLAS_INT32X4 vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

        if (N & 4) {
            *(int*)OutputC = _mm_cvtsi128_si32(vc);
            N -= 4;
            OutputC += 4;
            vc = _mm_shuffle_epi32(vc, _MM_SHUFFLE(0, 3, 2, 1));
        }

        uint32_t PackedValueC = (uint32_t)_mm_cvtsi128_si32(vc);
        for (size_t i = 0; i < N; ++i) {
            *((uint8_t*)OutputC + i) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}

#else

// Pure C++ implementation.
template<typename DataType, bool IsScalarB>
static
void
MlasQLinearMulKernel(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N
    )
{
    const float MinimumValue = (float)((int)std::numeric_limits<DataType>::min() - ZeroPointC);
    const float MaximumValue = (float)((int)std::numeric_limits<DataType>::max() - ZeroPointC);

    float ValueB;

    if (IsScalarB) {
        ValueB = ScaleB * (int32_t(InputB[0]) - ZeroPointB);
    }

    for (size_t n = 0; n < N; n++) {
        float ValueA = ScaleA * (int32_t(InputA[n]) - ZeroPointA);
        if (!IsScalarB) {
            ValueB = ScaleB * (int32_t(InputB[n]) - ZeroPointB);
        }
        float ValueC = (ValueA * ValueB) / ScaleC;
        ValueC = std::min(std::max(ValueC, MinimumValue), MaximumValue);
        OutputC[n] = (DataType)(int32_t)std::nearbyintf(ValueC + ZeroPointC);
    }
}

#endif

template<typename DataType>
void
MLASCALL
MlasQLinearMul(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t N,
    bool IsScalarB
    )
{
    afklhasdklf;
    if (IsScalarB) {
        MlasQLinearMulKernel<DataType, true>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    } else {
        MlasQLinearMulKernel<DataType, false>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}
