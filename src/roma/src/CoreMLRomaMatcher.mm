// CoreMLRomaMatcher.mm — Objective-C++ implementation backing the
// pure-C++ CoreMLRomaMatcher.hpp public header.
//
// MUST use MLComputeUnitsCPUAndGPU (see ai-models/README.md). ANE
// makes this model 4× slower because of grid_sample CPU↔ANE handoffs.

#include "av/roma/CoreMLRomaMatcher.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

// Focused OpenCV submodule headers — same ObjC++/stitching trap as
// MoGe + sphereDetection.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace av {
namespace roma {

struct CoreMLRomaMatcher::Impl
{
    MLModel* model = nil;
};

namespace {

// Write an OpenCV BGR image (already resized to kInputWidth × kInputHeight)
// into an [1,3,H,W] float32 NCHW MLMultiArray in [0,1] RGB.
void packNCHWFloat(const cv::Mat& bgr, MLMultiArray* arr)
{
    const NSInteger sC = [arr.strides[1] integerValue];
    const NSInteger sH = [arr.strides[2] integerValue];
    const NSInteger sW = [arr.strides[3] integerValue];
    float* dst = (float*)arr.dataPointer;

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32FC3, 1.0 / 255.0);

    for (int y = 0; y < kInputHeight; ++y) {
        const cv::Vec3f* row = rgbFloat.ptr<cv::Vec3f>(y);
        for (int x = 0; x < kInputWidth; ++x) {
            dst[0 * sC + y * sH + x * sW] = row[x][0];  // R
            dst[1 * sC + y * sH + x * sW] = row[x][1];  // G
            dst[2 * sC + y * sH + x * sW] = row[x][2];  // B
        }
    }
}

// Read an MLMultiArray of shape [1, C, H, W] float32 into a flat C*H*W
// std::vector<float>. Handles non-contiguous strides correctly.
void readNCHWFloat(MLMultiArray* arr, int C, int H, int W, std::vector<float>& out)
{
    out.resize(static_cast<size_t>(C) * H * W);
    const NSInteger sC = [arr.strides[1] integerValue];
    const NSInteger sH = [arr.strides[2] integerValue];
    const NSInteger sW = [arr.strides[3] integerValue];
    const float* src = (const float*)arr.dataPointer;
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                out[c * H * W + y * W + x] = src[c * sC + y * sH + x * sW];
            }
        }
    }
}

}  // namespace

CoreMLRomaMatcher::CoreMLRomaMatcher(const std::string& mlpackagePath)
    : _impl(std::make_unique<Impl>())
{
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:mlpackagePath.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];

        BOOL isDir = NO;
        BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:path
                                                           isDirectory:&isDir];
        if (!exists) {
            throw std::runtime_error("Roma model not found: " + mlpackagePath);
        }

        NSURL* loadURL = url;
        if (![path hasSuffix:@".mlmodelc"]) {
            NSError* err = nil;
            NSURL* compiled = [MLModel compileModelAtURL:url error:&err];
            if (err != nil || compiled == nil) {
                std::string msg = "Roma: MLModel compileModelAtURL failed: ";
                msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
                throw std::runtime_error(msg);
            }
            loadURL = compiled;
        }

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        // CRITICAL: do NOT use MLComputeUnitsAll — see ai-models/README.md.
        // TinyRoMa with .all is ~4× slower than CPU because of grid_sample
        // ANE handoffs. cpuAndGPU is the production target (~12 ms / pair
        // at 480×640 on M-series GPU).
        config.computeUnits = MLComputeUnitsCPUAndGPU;

        NSError* err = nil;
        _impl->model = [MLModel modelWithContentsOfURL:loadURL
                                         configuration:config
                                                 error:&err];
        if (err != nil || _impl->model == nil) {
            std::string msg = "Roma: MLModel load failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
            throw std::runtime_error(msg);
        }

        // Validate the expected I/O schema.
        NSDictionary* inputs = _impl->model.modelDescription.inputDescriptionsByName;
        NSDictionary* outputs = _impl->model.modelDescription.outputDescriptionsByName;
        for (NSString* name in @[ @"im_A", @"im_B" ]) {
            if (inputs[name] == nil) {
                throw std::runtime_error(
                    std::string("Roma: model missing input '") +
                    [name UTF8String] + "'");
            }
        }
        for (NSString* name in @[ @"coarse_flow", @"coarse_certainty",
                                  @"fine_flow", @"fine_certainty" ]) {
            if (outputs[name] == nil) {
                throw std::runtime_error(
                    std::string("Roma: model missing output '") +
                    [name UTF8String] + "'");
            }
        }
    }
}

CoreMLRomaMatcher::~CoreMLRomaMatcher() = default;

RomaMatch CoreMLRomaMatcher::match(const std::string& imageAPath,
                                   const std::string& imageBPath)
{
    cv::Mat bgrA = cv::imread(imageAPath, cv::IMREAD_COLOR);
    if (bgrA.empty()) {
        throw std::runtime_error("Roma: cv::imread failed: " + imageAPath);
    }
    cv::Mat bgrB = cv::imread(imageBPath, cv::IMREAD_COLOR);
    if (bgrB.empty()) {
        throw std::runtime_error("Roma: cv::imread failed: " + imageBPath);
    }

    const int origAW = bgrA.cols, origAH = bgrA.rows;
    const int origBW = bgrB.cols, origBH = bgrB.rows;

    cv::Mat resA, resB;
    cv::resize(bgrA, resA, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);
    cv::resize(bgrB, resB, cv::Size(kInputWidth, kInputHeight), 0, 0, cv::INTER_LINEAR);

    @autoreleasepool {
        NSError* err = nil;
        MLMultiArray* arrA = [[MLMultiArray alloc]
            initWithShape:@[ @1, @3, @(kInputHeight), @(kInputWidth) ]
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&err];
        if (err != nil || arrA == nil) {
            throw std::runtime_error("Roma: MLMultiArray(im_A) alloc failed");
        }
        MLMultiArray* arrB = [[MLMultiArray alloc]
            initWithShape:@[ @1, @3, @(kInputHeight), @(kInputWidth) ]
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&err];
        if (err != nil || arrB == nil) {
            throw std::runtime_error("Roma: MLMultiArray(im_B) alloc failed");
        }

        packNCHWFloat(resA, arrA);
        packNCHWFloat(resB, arrB);

        NSDictionary* feat = @{
            @"im_A": [MLFeatureValue featureValueWithMultiArray:arrA],
            @"im_B": [MLFeatureValue featureValueWithMultiArray:arrB],
        };
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:feat error:&err];
        if (err != nil) {
            throw std::runtime_error(std::string("Roma: provider init: ") +
                                     [[err localizedDescription] UTF8String]);
        }

        id<MLFeatureProvider> result =
            [_impl->model predictionFromFeatures:provider error:&err];
        if (err != nil || result == nil) {
            std::string msg = "Roma: prediction failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
            throw std::runtime_error(msg);
        }

        MLMultiArray* cFlow = [result featureValueForName:@"coarse_flow"].multiArrayValue;
        MLMultiArray* cCert = [result featureValueForName:@"coarse_certainty"].multiArrayValue;
        MLMultiArray* fFlow = [result featureValueForName:@"fine_flow"].multiArrayValue;
        MLMultiArray* fCert = [result featureValueForName:@"fine_certainty"].multiArrayValue;

        if (!cFlow || !cCert || !fFlow || !fCert) {
            throw std::runtime_error("Roma: output features missing");
        }

        RomaMatch out;
        out.origImageAWidth = origAW;
        out.origImageAHeight = origAH;
        out.origImageBWidth = origBW;
        out.origImageBHeight = origBH;

        readNCHWFloat(cFlow, 2, kCoarseHeight, kCoarseWidth, out.coarseFlow);
        readNCHWFloat(cCert, 1, kCoarseHeight, kCoarseWidth, out.coarseCertainty);
        readNCHWFloat(fFlow, 2, kFineHeight, kFineWidth, out.fineFlow);
        readNCHWFloat(fCert, 1, kFineHeight, kFineWidth, out.fineCertainty);

        return out;
    }
}

}  // namespace roma
}  // namespace av
