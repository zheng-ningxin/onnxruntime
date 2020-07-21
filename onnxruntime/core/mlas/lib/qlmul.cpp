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

#include "qladd.h"

#if defined(MLAS_NEON_INTRINSICS)


#if ! defined(_MSC_VER)

#define vld1q_s8_ex(pD, align) vld1q_s8((int8_t*)__builtin_assume_aligned(pD, ((align)/8)))
#define vst1_s8_ex(pD, D, align) vst1_s8((int8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vst1q_s8_ex(pD, D, align) vst1q_s8((int8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vld1q_u8_ex(pD, align) vld1q_u8((uint8_t*)__builtin_assume_aligned(pD, ((align)/8)))
#define vst1_u8_ex(pD, D, align) vst1_u8((uint8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vst1q_u8_ex(pD, D, align) vst1q_u8((uint8_t*)__builtin_assume_aligned(pD, ((align)/8)), D)
#define vst1_lane_u32_ex(pD, D, lane, align) vst1_lane_u32((uint32_t*)__builtin_assume_aligned(pD, ((align)/8)), D, lane)
#define vst1_lane_u16_ex(pD, D, lane, align) vst1_lane_u16((uint16_t*)__builtin_assume_aligned(pD, ((align)/8)), D, lane)

#endif

template <typename DataType>
class MLAS_SignedUnsignedIntOps;

template <>
class MLAS_SignedUnsignedIntOps<uint8_t>
{
public:
    typedef uint8_t T;
    typedef uint8x8_t i8x8_t;
    typedef uint8x16_t i8x16_t;
    typedef uint16x8_t i16x8_t;

    static MLAS_FORCEINLINE i8x8_t vmov_n_i8(T value)
    {
        return vmov_n_u8(value);
    }

    static MLAS_FORCEINLINE i8x8_t vget_low_i8(i8x16_t a)
    {
        return vget_low_u8(a);
    }

    static MLAS_FORCEINLINE i8x8_t vget_high_i8(i8x16_t a)
    {
        return vget_high_u8(a);
    }

    static MLAS_FORCEINLINE i16x8_t vsubl_i8(i8x8_t a, i8x8_t b)
    {
        return vsubl_u8(a, b);
    }

    static MLAS_FORCEINLINE int16x8_t vreinterpretq_s16_i16(i16x8_t a)
    {
        return vreinterpretq_s16_u16(a);
    }

    static MLAS_FORCEINLINE uint32x4_t vreinterpretq_u32_i8(i8x16_t a)
    {
        return vreinterpretq_u32_u8(a);
    }

    static MLAS_FORCEINLINE uint16x8_t vreinterpretq_u16_i8(i8x16_t a)
    {
        return vreinterpretq_u16_u8(a);
    }

    static MLAS_FORCEINLINE uint32x2_t vreinterpret_u32_i8(i8x8_t a)
    {
        return vreinterpret_u32_u8(a);
    }

    static MLAS_FORCEINLINE uint16x4_t vreinterpret_u16_i8(i8x8_t a)
    {
        return vreinterpret_u16_u8(a);
    }

    static MLAS_FORCEINLINE i8x16_t vld1q_i8(T const * ptr)
    {
        return vld1q_u8_ex(ptr, 8);
    }

    static MLAS_FORCEINLINE void vst1_i8(T* ptr, i8x8_t a)
    {
        vst1_u8_ex(ptr, a, 8);
    }

    static MLAS_FORCEINLINE void vst1q_i8(T* ptr, i8x16_t a)
    {
        vst1q_u8_ex(ptr, a, 8);
    }

    template <int n>
    static MLAS_FORCEINLINE void vst1_lane_i8(T* ptr, i8x8_t a)
    {
        vst1_lane_u8(ptr, a, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x16_t vextq_i8(i8x16_t lo, i8x16_t hi)
    {
        return vextq_u8(lo, hi, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x8_t vext_i8(i8x8_t lo, i8x8_t hi)
    {
        return vext_u8(lo, hi, n);
    }

    static MLAS_FORCEINLINE i8x16_t combine_i8_s16(int16x8_t v0, int16x8_t v1)
    {

#if defined(MLAS_NEON64_INTRINSICS)
         return vqmovun_high_s16(vqmovun_s16(v0), v1);
#else
         return vcombine_u8(vqmovun_s16(v0), vqmovun_s16(v1));
#endif

    }
};

template <>
class MLAS_SignedUnsignedIntOps<int8_t>
{
public:
    typedef int8_t T;
    typedef int8x8_t i8x8_t;
    typedef int8x16_t i8x16_t;
    typedef int16x8_t i16x8_t;

    static MLAS_FORCEINLINE i8x8_t vmov_n_i8(T value)
    {
        return vmov_n_s8(value);
    }

    static MLAS_FORCEINLINE i8x8_t vget_low_i8(i8x16_t a)
    {
        return vget_low_s8(a);
    }

    static MLAS_FORCEINLINE i8x8_t vget_high_i8(i8x16_t a)
    {
        return vget_high_s8(a);
    }

    static MLAS_FORCEINLINE i16x8_t vsubl_i8(i8x8_t a, i8x8_t b)
    {
        return vsubl_s8(a, b);
    }

    static MLAS_FORCEINLINE int16x8_t vreinterpretq_s16_i16(i16x8_t a)
    {
        return a;
    }

    static MLAS_FORCEINLINE uint32x4_t vreinterpretq_u32_i8(i8x16_t a)
    {
        return vreinterpretq_u32_s8(a);
    }

    static MLAS_FORCEINLINE uint16x8_t vreinterpretq_u16_i8(i8x16_t a)
    {
        return vreinterpretq_u16_s8(a);
    }

    static MLAS_FORCEINLINE uint32x2_t vreinterpret_u32_i8(i8x8_t a)
    {
        return vreinterpret_u32_s8(a);
    }

    static MLAS_FORCEINLINE uint16x4_t vreinterpret_u16_i8(i8x8_t a)
    {
        return vreinterpret_u16_s8(a);
    }

    static MLAS_FORCEINLINE i8x16_t vld1q_i8(T const * ptr)
    {
        return vld1q_s8_ex(ptr, 8);
    }

    static MLAS_FORCEINLINE void vst1_i8(T* ptr, i8x8_t a)
    {
        vst1_s8_ex(ptr, a, 8);
    }

    static MLAS_FORCEINLINE void vst1q_i8(T* ptr, i8x16_t a)
    {
        vst1q_s8_ex(ptr, a, 8);
    }

    template <int n>
    static MLAS_FORCEINLINE void vst1_lane_i8(T* ptr, i8x8_t a)
    {
        vst1_lane_s8(ptr, a, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x16_t vextq_i8(i8x16_t lo, i8x16_t hi)
    {
        return vextq_s8(lo, hi, n);
    }

    template <int n>
    static MLAS_FORCEINLINE i8x8_t vext_i8(i8x8_t lo, i8x8_t hi)
    {
        return vext_s8(lo, hi, n);
    }

    static MLAS_FORCEINLINE i8x16_t combine_i8_s16(int16x8_t v0, int16x8_t v1)
    {

#if defined(MLAS_NEON64_INTRINSICS)
         return vqmovn_high_s16(vqmovn_s16(v0), v1);
#else
         return vcombine_s8(vqmovn_s16(v0), vqmovn_s16(v1));
#endif

    }
};

#if defined(MLAS_NEON64_INTRINSICS)

#define MlasMoveHighS16S32(s16x8) vmovl_high_s16(s16x8)
#define MlasCombineS16S32(lo, hi) vqmovn_high_s32(vqmovn_s32(lo), hi)

#else

#define MlasMoveHighS16S32(s16x8) vmovl_s16(vget_high_s16(s16x8))
#define MlasCombineS16S32(lo, hi) vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))

#endif

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
    typedef MLAS_SignedUnsignedIntOps<DataType> SUI;

    const float ScaleRatio = ScaleA * ScaleB / ScaleC;
    const float32x4_t VectorScaleRatio = vld1q_dup_f32(&ScaleRatio);
    const typename SUI::i8x8_t VectorZeroPointA = SUI::vmov_n_i8((DataType)ZeroPointA);
    const typename SUI::i8x8_t VectorZeroPointB = SUI::vmov_n_i8((DataType)ZeroPointB);
    const int16x8_t VectorZeroPointC = vmovq_n_s16((int16_t)ZeroPointC);

    int16x8_t vb0_s16x8, vb1_s16x8;
    if (IsScalarB) {
        const typename SUI::i8x8_t VectorB0 = SUI::vmov_n_i8(*InputB);
        vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
        vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));
    }

    while (N >= 16) {
        const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(InputA);
        InputA += 16;
        const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
        const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));

        if (!IsScalarB) {
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(InputB);
            InputB += 16;
            vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));
        }
        
        int32x4_t vacc0_lo = vmull_s16(vget_low_s16(va0_s16x8), vget_low(vb0_s16x8));
        int32x4_t vacc0_hi = vmull_s16(vget_high_s16(va0_s16x8), vget_high_s16(vb0_s16x8));
        int32x4_t vacc1_lo = vmull_s16(vget_low_s16(va1_s16x8), vget_low(vb1_s16x8));
        int32x4_t vacc1_hi = vmull_s16(vget_high_s16(va1_s16x8), vget_high_s16(vb1_s16x8));
        vacc0_lo = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc_lo)));
        vacc0_hi = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc0_hi)));
        vacc1_lo = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc1_lo)));
        vacc1_hi = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc1_hi)));

        // Pack, saturate, and add output zero point.
        const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), VectorZeroPointC);
        const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), VectorZeroPointC);
        typename SUI::i8x16_t vc = SUI::combine_i8_s16(vacc0, vacc1);

        N -= 16;
        SUI::vst1q_i8(OutputC, vc);
        OutputC += 16;
    }

    if (N > 0) {
        typename SUI::T TailDataA[16] = { 0 };
        typename SUI::T TailDataB[16] = { 0 };

        MlasCopyTailBytes((uint8_t*)TailDataA, (const uint8_t*)InputA, N);
        if (!IsScalarB) {
            MlasCopyTailBytes((uint8_t*)TailDataB, (const uint8_t*)InputB, N);
        }

        const typename SUI::i8x16_t VectorA0 = SUI::vld1q_i8(TailDataA);
        const int16x8_t va0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorA0), VectorZeroPointA));
        const int16x8_t va1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorA0), VectorZeroPointA));

        if (!IsScalarB) {
            const typename SUI::i8x16_t VectorB0 = SUI::vld1q_i8(TailDataB);
            vb0_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_low_i8(VectorB0), VectorZeroPointB));
            vb1_s16x8 = SUI::vreinterpretq_s16_i16(SUI::vsubl_i8(SUI::vget_high_i8(VectorB0), VectorZeroPointB));
        }
        
        int32x4_t vacc0_lo = vmull_s16(vget_low_s16(va0_s16x8), vget_low(vb0_s16x8));
        int32x4_t vacc0_hi = vmull_s16(vget_high_s16(va0_s16x8), vget_high_s16(vb0_s16x8));
        int32x4_t vacc1_lo = vmull_s16(vget_low_s16(va1_s16x8), vget_low(vb1_s16x8));
        int32x4_t vacc1_hi = vmull_s16(vget_high_s16(va1_s16x8), vget_high_s16(vb1_s16x8));
        vacc0_lo = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc_lo)));
        vacc0_hi = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc0_hi)));
        vacc1_lo = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc1_lo)));
        vacc1_hi = vcvt_s32_f32(vmulq_f32(VectorScaleRatio, vcvt_f32_s32(vacc1_hi)));

        // Pack, saturate, and add output zero point.
        const int16x8_t vacc0 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi)), VectorZeroPointC);
        const int16x8_t vacc1 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi)), VectorZeroPointC);
        typename SUI::i8x16_t vc = SUI::combine_i8_s16(vacc0, vacc1);

        typename SUI::i8x8_t i8x8 = SUI::vget_low_i8(vc);
        if (N & 8) {
            SUI::vst1_i8(OutputC, i8x8);
            OutputC += 8;
            i8x8 = SUI::vget_high_i8(vc);
        }
        if (N & 4) {
            vst1_lane_u32_ex((uint32_t*)OutputC, SUI::vreinterpret_u32_i8(i8x8), 0, 8);
            OutputC += 4;
            i8x8 = SUI::template vext_i8<4>(i8x8, i8x8);
        }
        if (N & 2) {
            vst1_lane_u16_ex((uint16_t*)OutputC, SUI::vreinterpret_u16_i8(i8x8), 0, 8);
            OutputC += 2;
            i8x8 = SUI::template vext_i8<2>(i8x8, i8x8);
        }
        if (N & 1) {
            SUI::template vst1_lane_i8<0>(OutputC, i8x8);
        }
    }
}

#elif defined(MLAS_SSE2_INTRINSICS)

template <typename DataType>
MLAS_FORCEINLINE
static
MLAS_INT32X4
MlasShiftRightInt32(
    MLAS_INT32X4 v,
    int imm
    );

template<>
MLAS_INT32X4
MlasShiftRightInt32<int8_t>(
    MLAS_INT32X4 v,
    int imm
    )
{
    return _mm_srai_epi32(v, imm);
}

template<>
MLAS_INT32X4
MlasShiftRightInt32<uint8_t>(
    MLAS_INT32X4 v,
    int imm
    )
{
    return _mm_srli_epi32(v, imm);
}

template <typename DataType>
MLAS_FORCEINLINE
static
MLAS_INT32X4
MlasPackS16_128(
    MLAS_INT32X4 a,
    MLAS_INT32X4 b
    );

template <>
MLAS_INT32X4
MlasPackS16_128<uint8_t>(
    MLAS_INT32X4 a,
    MLAS_INT32X4 b
    )
{
    return _mm_packus_epi16(a, b);
}

template <>
MLAS_INT32X4
MlasPackS16_128<int8_t>(
    MLAS_INT32X4 a,
    MLAS_INT32X4 b
    )
{
    return _mm_packs_epi16(a, b);
}

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
        vb_lo = _mm_sub_epi32(MlasBroadcastInt32x4((int32_t)*InputB), VectorZeroPointB);
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

        auto r_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(va_lo), _mm_cvtepi32_ps(vb_lo)), VectorScaleRatio)), VectorZeroPointC);
        auto r_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(va_hi), _mm_cvtepi32_ps(vb_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_i16x8 = _mm_packs_epi32(r_lo, r_hi);
        auto vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);
        _mm_storel_epi64((MLAS_INT32X4*)OutputC, vc);
        OutputC += 8;
        N -= 8;
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

        auto r_lo = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(va_lo), _mm_cvtepi32_ps(vb_lo)), VectorScaleRatio)), VectorZeroPointC);
        auto r_hi = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_mul_ps(_mm_cvtepi32_ps(va_hi), _mm_cvtepi32_ps(vb_hi)), VectorScaleRatio)), VectorZeroPointC);
        const auto vc_i16x8 = _mm_packs_epi32(r_lo, r_hi);
        auto vc = MlasPackS16_128<DataType>(vc_i16x8, vc_i16x8);

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

template <typename DataType>
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
    if (IsScalarB) {
        MlasQLinearMulKernel<DataType, true>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    } else {
        MlasQLinearMulKernel<DataType, false>(
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, N);
    }
}

// Explicit instantiation
template
void
MlasQLinearMul<uint8_t>(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t N,
    bool IsScalarB
    );

template
void
MlasQLinearMul<int8_t>(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t N,
    bool IsScalarB
    );
