{
    "header": {
        "releaseVersion": "2025.1.0-develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "FeatureExtraction": "1.3",
            "ImageMatching": "2.0",
            "FeatureMatching": "2.0",
            "StructureFromMotion": "3.3",
            "PrepareDenseScene": "3.1",
            "DepthMap": "5.0",
            "DepthMapFilter": "4.0",
            "Meshing": "7.0",
            "Texturing": "6.0",
            "CopyFiles": "3.0"
        }
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [0, 0],
            "parallelization": { "blockSize": 0, "size": 1, "split": 1 },
            "uid": "1111111111111111111111111111111111111111",
            "inputs": {
                "viewpoints": [],
                "intrinsics": [],
                "sensorDatabase": "${ALICEVISION_SENSOR_DB}",
                "defaultFieldOfView": 45.0,
                "groupCameraFallback": "folder",
                "allowedCameraModels": ["pinhole", "radial1", "radial3", "brown", "fisheye4"],
                "rawColorInterpretation": "LibRawWhiteBalancing",
                "viewIdRegex": ".*?(\\d+)",
                "verboseLevel": "info"
            },
            "internalInputs": {
                "label": "Camera Init",
                "color": "#3F51B5"
            },
            "outputs": {
                "output": "{nodeCacheFolder}/cameraInit.sfm",
                "viewpointsOutput": "{nodeCacheFolder}/viewpoints.json"
            }
        },
        "FeatureExtraction_1": {
            "nodeType": "FeatureExtraction",
            "position": [200, 0],
            "parallelization": { "blockSize": 40, "size": 1, "split": 1 },
            "uid": "2222222222222222222222222222222222222222",
            "inputs": {
                "input": "{CameraInit_1.output}",
                "describerTypes": ["dspsift"],
                "describerPreset": "normal",
                "describerQuality": "normal",
                "contrastFiltering": "GridSort",
                "relativePeakThreshold": 0.01,
                "gridFiltering": true,
                "forceCpuExtraction": false,
                "maxThreads": 0,
                "verboseLevel": "info"
            },
            "outputs": {
                "output": "{nodeCacheFolder}"
            }
        },
        "ImageMatching_1": {
            "nodeType": "ImageMatching",
            "position": [400, 0],
            "parallelization": { "blockSize": 0, "size": 1, "split": 1 },
            "uid": "3333333333333333333333333333333333333333",
            "inputs": {
                "input": "{FeatureExtraction_1.input}",
                "featuresFolders": ["{FeatureExtraction_1.output}"],
                "method": "VocabularyTree",
                "tree": "${ALICEVISION_VOCTREE}",
                "minNbImages": 200,
                "maxDescriptors": 500,
                "nbMatches": 50,
                "verboseLevel": "info"
            },
            "outputs": {
                "output": "{nodeCacheFolder}/imageMatches.txt"
            }
        },
        "Meshing_1": {
            "nodeType": "Meshing",
            "position": [600, 0],
            "parallelization": { "blockSize": 0, "size": 1, "split": 1 },
            "uid": "4444444444444444444444444444444444444444",
            "inputs": {
                "input": "{ImageMatching_1.input}",
                "depthMapsFolder": "{ImageMatching_1.output}",
                "estimateSpaceFromSfM": true,
                "estimateSpaceMinObservations": 3,
                "estimateSpaceMinObservationAngle": 10.0,
                "maxInputPoints": 50000000,
                "maxPoints": 5000000,
                "maxPointsPerVoxel": 1000000,
                "minStep": 2,
                "partitioning": "singleBlock",
                "repartition": "multiResolution",
                "angleFactor": 15.0,
                "saveRawDensePointCloud": false,
                "voteFilteringForWeaklySupportedSurfaces": true,
                "advanced": {
                    "helperPointsGridSize": 10,
                    "densify": false,
                    "densifyNbFront": 1,
                    "densifyNbBack": 1,
                    "densifyScale": 20.0
                },
                "verboseLevel": "info"
            },
            "outputs": {
                "outputMesh": "{nodeCacheFolder}/mesh.obj",
                "output": "{nodeCacheFolder}/densePointCloud.abc"
            }
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [800, 0],
            "parallelization": { "blockSize": 0, "size": 1, "split": 1 },
            "uid": "5555555555555555555555555555555555555555",
            "inputs": {
                "inputFiles": [
                    {
                        "input": "{Meshing_1.outputMesh}"
                    }
                ],
                "outputFolder": "",
                "verboseLevel": "info"
            },
            "outputs": {
                "output": "{nodeCacheFolder}"
            }
        }
    }
}
