// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "MeshroomNative",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "ProjectModel", targets: ["ProjectModel"]),
        .executable(name: "MeshroomNativeApp", targets: ["MeshroomNativeApp"]),
    ],
    targets: [
        .target(name: "ProjectModel"),
        .executableTarget(
            name: "MeshroomNativeApp",
            dependencies: ["ProjectModel"],
            path: "Sources/App"
        ),
        .testTarget(
            name: "ProjectModelTests",
            dependencies: ["ProjectModel"],
            resources: [
                .copy("Fixtures"),
            ]
        ),
        .testTarget(
            name: "AppTests",
            dependencies: ["MeshroomNativeApp", "ProjectModel"],
            resources: [
                .copy("Fixtures"),
            ]
        ),
    ]
)
