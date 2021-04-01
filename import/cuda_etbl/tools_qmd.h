/*
 * Copyright 1993-2012 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _tools_qmd_h_
#define _tools_qmd_h_

typedef struct CUtoolsQMDV01_07_st
{
    /* DWORDS 0, 1
        OuterPut                        :31
        OuterOverflow                   :1
        OuterGet                        :31
        OuterStickyOverflow             :1
    */
    NvU32 OuterPut;
    NvU32 OuterGet;

    /* DWORDS 2, 3
        InnerGet                        :31
        InnerOverflow                   :1
        InnerPut                        :31
        InnerStickyOverflow             :1
    */
    NvU32 InnerGet;
    NvU32 InnerPut;

    /* DWORD 4 */
    NvU32 QmdReservedAA;

    /* DWORD 5 */
    NvU32 DependentQmdPointer;

    /* DWORD 6
        QmdGroupId                      :6
        QmdReservedA                    :2
        IsQueue                         :1
        AddToHeadOfQmdGroupLinkedList   :1
        SemaphoreReleaseEnable0         :1
        SemaphoreReleaseEnable1         :1
        RequireSchedulingPcas           :1
        DependentQmdScheduleEnable      :1
        DependentQmdType                :1
            QUEUE = 0
            GRID  = 1
        DependentQmdFieldCopy           :1
        QmdReservedB                    :16
    */
    NvU32 QmdSchedulingState0;

    /* DWORD 7
        CircularQueueSize               :25
        QmdReservedC                    :1
        InvalidateTextureHeaderCache    :1
        InvalidateTextureSamplerCache   :1
        InvalidateTextureDataCache      :1
        InvalidateShaderDataCache       :1
        InvalidateInstructionCache      :1
        InvalidateShaderConstantCache   :1
    */
    NvU32 QmdQueueAndCacheState0;

    /* DWORD 8 */
    NvU32 ProgramOffset;

    /* DWORD 9 */
    NvU32 CircularQueueAddrLower;

    /* DWORD 10
        CircularQueueAddrUpper          :8
        QmdReservedD                    :8
        CircularQueueEntrySize          :16
    */
    NvU32 CircularQueueState0;

    /* DWORD 11
        CwdReferenceCountId             :6
        CwdReferenceCountDeltaMinusOne  :8
        ReleaseMembarType               :1
            FE_NONE      = 0
            FE_SYSMEMBAR = 1
        CwdReferenceCountIncrEnable     :1
        CwdMembarType                   :2
            L1_NONE      = 0
            L1_SYSMEMBAR = 1
            L1_MEMBAR    = 3
        SequentiallyRunCtas             :1
        CwdReferenceCountDecrEnable     :1
        Throttled                       :1
        QmdReservedE2                   :3
        Fp32NanBehavior                 :1
            LEGACY          = 0
            FP64_COMPATIBLE = 1
        Fp32F2iNanaBehavior             :1
            PASS_ZERO       = 0
            PASS_INDEFINITE = 1
        ApiVisibleCallLimit             :1
            _32      = 0
            NO_CHECK = 1
        SharedMemoryBankMapping         :1
            FOUR_BYTES_PER_BANK  = 0
            EIGHT_BYTES_PER_BANK = 1
        QmdReservedE3                   :2
        SamplerIndex                    :1
            INDEPENDENTLY    = 0
            VIA_HEADER_INDEX = 1
        Fp32NarrowInstruction           :1
            UNUSED
    */
    NvU32 QmdCwdState0;

    /* DWORDS 12, 13
        CtaRasterWidth                  :32
        CtaRasterHeight                 :16
        CtaRasterDepth                  :16
    */
    NvU32 CtaRasterWidth;
    NvU32 CtaRasterHeightDepth;

    /* DWORDS 14, 15
        CtaRasterWidthResume            :32
        CtaRasterHeightResume           :16
        CtaRasterDepthResume            :16
    */
    NvU32 CtaRasterWidthResume;
    NvU32 CtaRasterHeightDepthResume;

    /* DWORD 16
        QueueEntriesPerCtaMinusOne      :7
        QmdReservedF1                   :3
        CoalesceWaitingPeriod           :8
        QmdReservedF2                   :14
    */
    NvU32 QmdQueueCoalescingState0;

    /* DWORD 17
        SharedMemorySize                :18
        QmdReservedG                    :14
    */
    NvU32 SharedMemorySize;

    /* DWORD 18
        QmdVersion                      :4
        QmdMajorVersion                 :4
        QmdReservedH                    :8
        CtaThreadDimension0             :16
    */
    NvU32 QmdVersionAndCtaThreadDim0;

    /* DWORD 19
        CtaThreadDimension1             :16
        CtaThreadDimension2             :16
    */
    NvU32 CtaThreadHeightDepthDimension;

    /* DWORD 20
        ConstantBufferValid[8]          :8
        QmdReservedI                    :21
        L1Configuration                 :3
            DIRECTLY_ADDRESSABLE_MEMORY_SIZE_16KB = 1
            DIRECTLY_ADDRESSABLE_MEMORY_SIZE_32KB = 2
            DIRECTLY_ADDRESSABLE_MEMORY_SIZE_48KB = 3
    */
    NvU32 ConstBuffersValidAndL1Config;

    /* DWORD 21 */
    NvU32 SmDisableMaskLower;
    NvU32 SmDisableMaskUpper;

    /* DWORDS 23, 24, 25
        Release0AddressLower            :32
        Release0AddressUpper            :8
        QmdReservedJ                    :8
        QmdReservedJ1                   :4
        Release0ReductionOp             :3
            ADD = 0
            MIN = 1
            MAX = 2
            INC = 3
            DEC = 4
            AND = 5
            OR  = 6
            XOR = 7
        QmdReservedK                    :1
        Release0ReductionFormat         :2
            UNSIGNED_32 = 0
            SIGNED_32   = 1
        Release0ReductionEnable         :1
        QmdReservedK1                   :4
        Release0StructureSize           :1
            FOUR_WORDS = 0
            ONE_WORD   = 1
        Release0Payload                 :32
    */
    NvU32 Release0AddressLower;
    NvU32 Release0Config;
    NvU32 Release0Payload;

    /* DWORDS 26, 27, 28
        Release1AddressLower            :32
        Release1AddressUpper            :8
        QmdReservedJ                    :8
        QmdReservedJ1                   :4
        Release1ReductionOp             :3
            ADD = 0
            MIN = 1
            MAX = 2
            INC = 3
            DEC = 4
            AND = 5
            OR  = 6
            XOR = 7
        QmdReservedK                    :1
        Release1ReductionFormat         :2
            UNSIGNED_32 = 0
            SIGNED_32   = 1
        Release1ReductionEnable         :1
        QmdReservedK1                   :4
        Release1StructureSize           :1
            FOUR_WORDS = 0
            ONE_WORD   = 1
        Release1Payload                 :32
    */
    NvU32 Release1AddressLower;
    NvU32 Release1Config;
    NvU32 Release1Payload;

    /* DWORDS 29 ... 44
        ConstantBufferAddrLower[i]      :32
        ConstantBufferAddrUpper[i]      :8
        ConstantBufferReservedAddr[i]   :6
        ConstantBufferInvalidate[i]     :1
        ConstantBufferSize[i]           :17
    */
    NvU32 ConstantBufferState[16];

    /* DWORD 45
        ShaderLocalMemoryLowSize        :24
        QmdReservedN                    :3
        BarrierCount                    :5
    */
    NvU32 LmemLowSizeAndBarrierCount;

    /* DWORD 46
        ShaderLocalMemoryHighSize       :24
        RegisterCount                   :8
    */
    NvU32 LmemHighSizeAndRegisterCount;

    /* DWORD 47
        ShaderLocalMemoryCrsSize        :24
        SassVersion                     :8
    */
    NvU32 CrsSizeAndSassVersion;

    /* DWORD 48
        HwOnlyInnerGet                  :31
        HwOnlyRequireSchedulingPcas     :1
    */
    NvU32 HwOnlyQmdData0;

    /* DWORD 49
        HwOnlyInnerPut                  :31
        QmdReservedP                    :1
    */
    NvU32 HwOnlyQmdData1;

    /* DWORD 50
        HwOnlySpanListHeadIndex         :30
        QmdReservedQ                    :1
        HwOnlySpanListHeadIndexValid    :1
    */
    NvU32 HwOnlyQmdData2;

    /* DWORD 51
        HwOnlySkedNextQmdPointer        :32
    */
    NvU32 HwOnlyQmdData3;

    /* DWORDS 52 ... 61 */
    NvU32 QmdSpareE;
    NvU32 QmdSpareF;
    NvU32 QmdSpareG;
    NvU32 QmdSpareH;
    NvU32 QmdSpareI;
    NvU32 QmdSpareJ;
    NvU32 QmdSpareK;
    NvU32 QmdSpareL;
    NvU32 QmdSpareM;
    NvU32 QmdSpareN;

    /* DWORDS 62, 63 */
    NvU32 DebugIdUpper;
    NvU32 DebugIdLower;
} CUtoolsQMDV01_07;

#endif //_tools_qmd_h_
