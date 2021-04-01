#ifndef __cuda_packing_h__
#define __cuda_packing_h__

#include "cuda_stdint.h"

#if defined(__LP64__) || defined(_WIN64)
// 64-bit
#define CU_32_BIT_PAD_ON_32_BIT_BUILDS(name)
#define CU_32_BIT_PAD_ON_64_BIT_BUILDS(name) uint32_t name;
#else
// 32-bit
#define CU_32_BIT_PAD_ON_32_BIT_BUILDS(name) uint32_t name;
#define CU_32_BIT_PAD_ON_64_BIT_BUILDS(name)
#endif

#if !defined(CU_SPECIFY_CUSTOM_PACKING_MACROS)

// DJGPP requires opting-in for #pragma pack(push/pop) support
#ifdef __DJGPP__
#define HANDLE_PRAGMA_PACK_PUSH_POP 1
#endif

// CU_PACK:
// Cross platform macro for defining packed structures.
// With begin/end pairs, it's possible to forget the end,
// and everything will still compile, but great amounts
// of trouble will ensue.  This single-macro approach is
// safer because forgetting the end-paren will yield an
// obvious compiler error.  Wrap struct definitions like
// this:
//
//   CU_PACK( struct foo {...}; )
//       or
//   CU_PACK( typedef struct foo_st {...} foo; )
//
// It is preferable to wrap individual struct definitions
// rather than entire files because for some compilers
// CU_PACK is defined using attribute syntax (which
// must be applied directly to each struct definition).

#ifdef _MSC_VER // MSVC
    #define CU_PACK(_struct_definition) \
        __pragma(pack(push, 1)) \
        _struct_definition \
        __pragma(pack(pop)) \

        // Note:  Do not remove preceding blank line!
#else
    #if defined(__GNUC__) && (__GNUC__ < 3)  // Old versions of GCC
        #define CU_PACK(_struct_definition) \
            __attribute__((packed)) \
            _struct_definition
    #else // GCC, DJGPP, anything that supports C99
        #define CU_PACK(_struct_definition) \
            _Pragma("pack(push, 1)") \
            _struct_definition \
            _Pragma("pack(pop)") \

        // Note:  Do not remove preceding blank line!
    #endif
#endif

// Implementation notes:
//
// #pragma is not usable within preprocessor macros, but
// every modern compiler supports a macro-compatible way
// to specify pragmas.  MSVC uses __pragma(...) and C99
// uses _Pragma("...").  Also, the original attempt to
// implement this had the packing size (hard-coded to 1
// above) as a parameter.  Unfortunately, GCC fails to
// preprocess in the correct order, so stringizing and
// string literal concatenation don't work for the
// parameter to _Pragma, preventing anything but hard-
// coded parameter strings.  Feel free to add another
// macro here for defining custom packing if you can
// find a well-supported way to do it.  Also, do not
// remove the blank lines in these macros!  They are
// protecting code that comes after the CU_PACK macros
// from ending up on the same line as macro-compatible
// pragmas, which may be bad on some compilers.

#endif // !defined(CU_SPECIFY_CUSTOM_PACKING_MACROS)

#endif // file guard
