// CoreMLMoGeRunner.mm — Objective-C++ implementation backing the
// pure-C++ CoreMLMoGeRunner.hpp public header.
//
// MoGe-2 takes NCHW float32 [1,3,504,672] in [0,1]. We:
//   1. cv::imread the input (BGR uint8)
//   2. cvtColor BGR → RGB
//   3. resize to 672×504 (INTER_LINEAR)
//   4. convertTo float32 with scale 1/255 (→ values in [0,1])
//   5. pack into an MLMultiArray NCHW (channels-first)
//   6. predictionFromFeatures
//   7. extract `points` Z-component, multiply by `metric_scale` → depth
//   8. extract `normal` xyz → unit-length normals
//   9. extract `mask` → uint8 0/1
//
// CoreMLSphereDetector's letterbox-avoidance comment also applies here:
// MoGe was trained with a fixed 672×504 letterboxed input pipeline, but
// the user's converted model has the resize baked in via plain resize.
// Direct cv::resize matches the conversion script's preprocessing.

#include "av/moge/CoreMLMoGeRunner.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// Focused OpenCV submodule headers — the umbrella opencv2/opencv.hpp
// pulls in opencv2/stitching/* which conflicts with ObjC NSObject
// symbols (`NO`, `seam_finder`, etc.). Same gotcha as
// src/sphere_detection/src/CoreMLSphereDetector.mm.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace av {
namespace moge {

struct CoreMLMoGeRunner::Impl
{
    MLModel* model = nil;
};

namespace {

// Read a NHWC float multi-array's [n][h][w][c] sample. CoreML
// MLMultiArrays expose strides as an NSArray<NSNumber*> in elements
// (not bytes); use them so we don't assume contiguity.
inline float sampleNHWC(MLMultiArray* arr, NSInteger n, NSInteger h,
                        NSInteger w, NSInteger c)
{
    const NSInteger sN = [arr.strides[0] integerValue];
    const NSInteger sH = [arr.strides[1] integerValue];
    const NSInteger sW = [arr.strides[2] integerValue];
    const NSInteger sC = [arr.strides[3] integerValue];
    const NSInteger off = n * sN + h * sH + w * sW + c * sC;
    return ((float*)arr.dataPointer)[off];
}

// Read a NHW float multi-array (no channel axis) — used for `mask`.
inline float sampleNHW(MLMultiArray* arr, NSInteger n, NSInteger h, NSInteger w)
{
    const NSInteger sN = [arr.strides[0] integerValue];
    const NSInteger sH = [arr.strides[1] integerValue];
    const NSInteger sW = [arr.strides[2] integerValue];
    const NSInteger off = n * sN + h * sH + w * sW;
    return ((float*)arr.dataPointer)[off];
}

}  // namespace

CoreMLMoGeRunner::CoreMLMoGeRunner(const std::string& mlpackagePath)
    : _impl(std::make_unique<Impl>())
{
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:mlpackagePath.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];

        BOOL isDir = NO;
        BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:path
                                                           isDirectory:&isDir];
        if (!exists) {
            throw std::runtime_error("MoGe model not found: " + mlpackagePath);
        }

        NSURL* loadURL = url;
        if (![path hasSuffix:@".mlmodelc"]) {
            NSError* err = nil;
            NSURL* compiled = [MLModel compileModelAtURL:url error:&err];
            if (err != nil || compiled == nil) {
                std::string msg = "MoGe: MLModel compileModelAtURL failed: ";
                msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
                throw std::runtime_error(msg);
            }
            loadURL = compiled;
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        NSError* err = nil;
        _impl->model = [MLModel modelWithContentsOfURL:loadURL
                                         configuration:config
                                                 error:&err];
        if (err != nil || _impl->model == nil) {
            std::string msg = "MoGe: MLModel load failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
            throw std::runtime_error(msg);
        }

        // Validate the expected input/output schema upfront so a
        // mis-converted model surfaces a clear error instead of silent
        // mis-decoded floats later.
        NSDictionary* inputs = _impl->model.modelDescription.inputDescriptionsByName;
        NSDictionary* outputs = _impl->model.modelDescription.outputDescriptionsByName;
        if (inputs[@"image"] == nil) {
            throw std::runtime_error("MoGe: model missing 'image' input");
        }
        for (NSString* name in @[ @"points", @"normal", @"mask", @"metric_scale" ]) {
            if (outputs[name] == nil) {
                throw std::runtime_error(
                    std::string("MoGe: model missing output '") +
                    [name UTF8String] + "'");
            }
        }
    }
}

CoreMLMoGeRunner::~CoreMLMoGeRunner() = default;

MoGeResult CoreMLMoGeRunner::predict(const std::string& imagePath)
{
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("MoGe: cv::imread failed: " + imagePath);
    }
    const int origW = bgr.cols;
    const int origH = bgr.rows;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized,
               cv::Size(kModelWidth, kModelHeight),
               0, 0, cv::INTER_LINEAR);

    cv::Mat resizedFloat;
    resized.convertTo(resizedFloat, CV_32FC3, 1.0 / 255.0);

    @autoreleasepool {
        // Build the [1, 3, 504, 672] NCHW input.
        NSError* err = nil;
        MLMultiArray* input =
            [[MLMultiArray alloc] initWithShape:@[ @1, @3, @(kModelHeight), @(kModelWidth) ]
                                       dataType:MLMultiArrayDataTypeFloat32
                                          error:&err];
        if (err != nil || input == nil) {
            throw std::runtime_error("MoGe: MLMultiArray alloc failed");
        }

        const NSInteger sC = [input.strides[1] integerValue];
        const NSInteger sH = [input.strides[2] integerValue];
        const NSInteger sW = [input.strides[3] integerValue];
        float* dst = (float*)input.dataPointer;
        for (int y = 0; y < kModelHeight; ++y) {
            const cv::Vec3f* row = resizedFloat.ptr<cv::Vec3f>(y);
            for (int x = 0; x < kModelWidth; ++x) {
                // Channels-first (NCHW): write R, G, B into channels 0, 1, 2.
                dst[0 * sC + y * sH + x * sW] = row[x][0];
                dst[1 * sC + y * sH + x * sW] = row[x][1];
                dst[2 * sC + y * sH + x * sW] = row[x][2];
            }
        }

        NSDictionary* feat = @{ @"image": [MLFeatureValue featureValueWithMultiArray:input] };
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:feat error:&err];
        if (err != nil) {
            throw std::runtime_error(std::string("MoGe: provider init: ") +
                                     [[err localizedDescription] UTF8String]);
        }

        id<MLFeatureProvider> result =
            [_impl->model predictionFromFeatures:provider error:&err];
        if (err != nil || result == nil) {
            std::string msg = "MoGe: prediction failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
            throw std::runtime_error(msg);
        }

        MLMultiArray* points = [result featureValueForName:@"points"].multiArrayValue;
        MLMultiArray* normal = [result featureValueForName:@"normal"].multiArrayValue;
        MLMultiArray* mask = [result featureValueForName:@"mask"].multiArrayValue;
        MLMultiArray* metricScaleArr = [result featureValueForName:@"metric_scale"].multiArrayValue;

        if (points == nil || normal == nil || mask == nil || metricScaleArr == nil) {
            throw std::runtime_error("MoGe: output features missing");
        }

        const float metricScale = ((float*)metricScaleArr.dataPointer)[0];

        MoGeResult out;
        out.width = kModelWidth;
        out.height = kModelHeight;
        out.origImageWidth = origW;
        out.origImageHeight = origH;
        out.metricScale = metricScale;
        out.depthMeters.resize(static_cast<size_t>(kModelWidth) * kModelHeight);
        out.normalXYZ.resize(static_cast<size_t>(kModelWidth) * kModelHeight * 3);
        out.mask.resize(static_cast<size_t>(kModelWidth) * kModelHeight);

        for (int y = 0; y < kModelHeight; ++y) {
            for (int x = 0; x < kModelWidth; ++x) {
                const size_t pix = static_cast<size_t>(y) * kModelWidth + x;

                // points: [1, 504, 672, 3] — Z is forward in MoGe's frame.
                const float z = sampleNHWC(points, 0, y, x, 2);
                out.depthMeters[pix] = z * metricScale;

                const float nx = sampleNHWC(normal, 0, y, x, 0);
                const float ny = sampleNHWC(normal, 0, y, x, 1);
                const float nz = sampleNHWC(normal, 0, y, x, 2);
                out.normalXYZ[pix * 3 + 0] = nx;
                out.normalXYZ[pix * 3 + 1] = ny;
                out.normalXYZ[pix * 3 + 2] = nz;

                const float m = sampleNHW(mask, 0, y, x);
                out.mask[pix] = (m > 0.5f) ? 1 : 0;
            }
        }

        return out;
    }
}

}  // namespace moge
}  // namespace av
