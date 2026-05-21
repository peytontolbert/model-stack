// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "ModelStackMetal",
    platforms: [
        .iOS(.v15),
        .macOS(.v13)
    ],
    products: [
        .library(name: "ModelStackMetal", targets: ["ModelStackMetal"])
    ],
    targets: [
        .target(name: "ModelStackMetal"),
        .testTarget(
            name: "ModelStackMetalTests",
            dependencies: ["ModelStackMetal"]
        )
    ]
)
