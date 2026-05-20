// This is the *single* translation unit that emits metal-cpp's
// out-of-line definitions. Every other TU includes the headers
// without the _PRIVATE_IMPLEMENTATION macros.
//
// Per Apple's metal-cpp README:
//   "Define these macros in exactly one translation unit."

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
