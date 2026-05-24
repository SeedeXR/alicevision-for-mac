// CoreMLSphereDetector.mm — Objective-C++ implementation backing the
// pure-C++ CoreMLSphereDetector.hpp public header.
//
// The upstream model is an Apple Vision-style YOLOv8n export:
//   inputs: image (RGB 640x640), iouThreshold (double), confidenceThreshold (double)
//   outputs: coordinates (Nx4, normalized cx,cy,w,h), confidence (NxC)
// NMS is baked into the model graph so we don't run it ourselves.
//
// Image preprocessing here uses a simple cv::resize to 640x640. The
// upstream sphere-detection use case is calibration shots where the
// sphere fills a large fraction of the frame, so the aspect-ratio
// distortion from a direct resize is acceptable. Switch to a letterbox
// preprocessor if a future model is trained on padded inputs.

#include "av/sphere/CoreMLSphereDetector.hpp"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

// Use focused OpenCV submodule headers — the umbrella opencv2/opencv.hpp
// pulls in opencv2/stitching/* which defines identifiers that clash with
// macOS SDK Objective-C symbols (`seam_finder`, `NO`, etc.).
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace av {
namespace sphere {

namespace {

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;
constexpr double kIoUThreshold = 0.5;

// Pixel format: 32-bit BGRA. CoreML accepts this and converts to RGB
// internally when the model's input is declared as colorSpace=RGB.
constexpr OSType kPixelFormat = kCVPixelFormatType_32BGRA;

CVPixelBufferRef makePixelBuffer(const cv::Mat& bgra)
{
    CVPixelBufferRef buffer = nullptr;
    NSDictionary* attrs = @{
        (id)kCVPixelBufferIOSurfacePropertiesKey: @{},
    };
    CVReturn rc = CVPixelBufferCreate(kCFAllocatorDefault,
                                      bgra.cols, bgra.rows,
                                      kPixelFormat,
                                      (__bridge CFDictionaryRef)attrs,
                                      &buffer);
    if (rc != kCVReturnSuccess || buffer == nullptr) {
        throw std::runtime_error("CVPixelBufferCreate failed");
    }
    CVPixelBufferLockBaseAddress(buffer, 0);
    uint8_t* dst = (uint8_t*)CVPixelBufferGetBaseAddress(buffer);
    const size_t stride = CVPixelBufferGetBytesPerRow(buffer);
    for (int y = 0; y < bgra.rows; ++y) {
        memcpy(dst + y * stride,
               bgra.ptr<uchar>(y),
               static_cast<size_t>(bgra.cols) * 4);
    }
    CVPixelBufferUnlockBaseAddress(buffer, 0);
    return buffer;
}

}  // namespace

struct CoreMLSphereDetector::Impl
{
    MLModel* model = nil;
    NSString* coordinatesOutputName = nil;
    NSString* confidenceOutputName = nil;
};

CoreMLSphereDetector::CoreMLSphereDetector(const std::string& mlpackagePath)
    : _impl(std::make_unique<Impl>())
{
    @autoreleasepool {
        NSString* path = [NSString stringWithUTF8String:mlpackagePath.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];

        // Compile if the path points to an uncompiled .mlpackage / .mlmodel.
        // CoreML accepts a compiled .mlmodelc URL directly; for raw packages
        // we ask the framework to compile to a temp dir.
        NSURL* loadURL = url;
        BOOL isDir = NO;
        BOOL exists = [[NSFileManager defaultManager] fileExistsAtPath:path
                                                          isDirectory:&isDir];
        if (!exists) {
            throw std::runtime_error("model not found: " + mlpackagePath);
        }
        if (![path hasSuffix:@".mlmodelc"]) {
            NSError* err = nil;
            NSURL* compiled = [MLModel compileModelAtURL:url error:&err];
            if (err != nil || compiled == nil) {
                std::string msg = "MLModel compileModelAtURL failed: ";
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
            std::string msg = "MLModel load failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
            throw std::runtime_error(msg);
        }

        // Cache output names — Vision-style YOLOv8 calls them
        // "coordinates" and "confidence".
        for (NSString* name in _impl->model.modelDescription.outputDescriptionsByName) {
            if ([name isEqualToString:@"coordinates"]) {
                _impl->coordinatesOutputName = name;
            } else if ([name isEqualToString:@"confidence"]) {
                _impl->confidenceOutputName = name;
            }
        }
        if (_impl->coordinatesOutputName == nil || _impl->confidenceOutputName == nil) {
            throw std::runtime_error(
                "model does not expose Vision-style 'coordinates' + 'confidence' outputs");
        }
    }
}

CoreMLSphereDetector::~CoreMLSphereDetector() = default;

DetectionResult CoreMLSphereDetector::predict(const std::string& imagePath,
                                              float minScore)
{
    cv::Mat bgr = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("cv::imread failed for: " + imagePath);
    }
    const int origW = bgr.cols;
    const int origH = bgr.rows;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(kInputWidth, kInputHeight), 0, 0,
               cv::INTER_LINEAR);
    cv::Mat bgra;
    cv::cvtColor(resized, bgra, cv::COLOR_BGR2BGRA);

    @autoreleasepool {
        CVPixelBufferRef pb = makePixelBuffer(bgra);

        NSError* err = nil;
        MLFeatureValue* imageValue =
            [MLFeatureValue featureValueWithPixelBuffer:pb];
        MLFeatureValue* iouValue =
            [MLFeatureValue featureValueWithDouble:kIoUThreshold];
        MLFeatureValue* confValue =
            [MLFeatureValue featureValueWithDouble:static_cast<double>(minScore)];

        NSDictionary* dict = @{
            @"image": imageValue,
            @"iouThreshold": iouValue,
            @"confidenceThreshold": confValue,
        };
        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict
                                                              error:&err];
        if (err != nil) {
            CVPixelBufferRelease(pb);
            std::string msg = "MLDictionaryFeatureProvider failed: ";
            msg += [[err localizedDescription] UTF8String];
            throw std::runtime_error(msg);
        }

        id<MLFeatureProvider> result =
            [_impl->model predictionFromFeatures:provider error:&err];
        CVPixelBufferRelease(pb);
        if (err != nil || result == nil) {
            std::string msg = "MLModel predictionFromFeatures failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "(no detail)";
            throw std::runtime_error(msg);
        }

        MLMultiArray* coords =
            [result featureValueForName:_impl->coordinatesOutputName].multiArrayValue;
        MLMultiArray* conf =
            [result featureValueForName:_impl->confidenceOutputName].multiArrayValue;
        if (coords == nil || conf == nil) {
            throw std::runtime_error("model output missing coordinates/confidence");
        }

        // Vision YOLOv8 NMS output:
        //   coordinates : [N, 4]  (cx, cy, w, h) normalized to [0, 1]
        //   confidence  : [N, num_classes]   (already filtered by confidenceThreshold)
        NSArray<NSNumber*>* coordShape = coords.shape;
        NSArray<NSNumber*>* confShape = conf.shape;
        if (coordShape.count < 2 || confShape.count < 2) {
            return DetectionResult{ {}, origW, origH };
        }
        const NSInteger n = [coordShape[0] integerValue];
        const NSInteger numClasses = [confShape[1] integerValue];
        if (n == 0) {
            return DetectionResult{ {}, origW, origH };
        }

        DetectionResult out;
        out.imageWidth = origW;
        out.imageHeight = origH;
        out.detections.reserve(static_cast<size_t>(n));

        for (NSInteger i = 0; i < n; ++i) {
            const double cx = [coords[@[@(i), @(0)]] doubleValue];
            const double cy = [coords[@[@(i), @(1)]] doubleValue];
            const double w = [coords[@[@(i), @(2)]] doubleValue];
            const double h = [coords[@[@(i), @(3)]] doubleValue];

            double maxScore = 0.0;
            for (NSInteger c = 0; c < numClasses; ++c) {
                const double s = [conf[@[@(i), @(c)]] doubleValue];
                if (s > maxScore) maxScore = s;
            }
            if (maxScore < static_cast<double>(minScore)) {
                continue;
            }

            Detection det;
            det.x1 = static_cast<float>((cx - w * 0.5) * origW);
            det.y1 = static_cast<float>((cy - h * 0.5) * origH);
            det.x2 = static_cast<float>((cx + w * 0.5) * origW);
            det.y2 = static_cast<float>((cy + h * 0.5) * origH);
            det.score = static_cast<float>(maxScore);
            out.detections.push_back(det);
        }

        // Sort highest score first (callers pick detection[0]).
        std::sort(out.detections.begin(), out.detections.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.score > b.score;
                  });
        return out;
    }
}

}  // namespace sphere
}  // namespace av
