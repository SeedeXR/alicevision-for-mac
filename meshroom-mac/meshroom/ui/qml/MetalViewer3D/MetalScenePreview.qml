/*
 * MetalScenePreview
 *
 * Mac-native QtQuick3D-based 3D preview viewer for the SegmentationBiRefNet/
 * ScenePreview pipeline. Replaces the Qt3D Scene3D-based `Viewer3D` which
 * crashes on macOS 26.5 + Qt 6.11.1 + Fusion style because Scene3D can't
 * share the Metal RHI context with QtQuick.
 *
 * Design rationale (memory/philosophy.md):
 *
 *   - UMA-first: QtQuick3D's RHI backend is Metal on macOS. Vertex / index
 *     / texture buffers are uploaded once into MTLResourceStorageModeShared
 *     and read directly by the GPU — no host↔device copies.
 *   - Direct Metal use first, abstraction second: QtQuick3D View3D dispatches
 *     to Metal RHI without intermediate OpenGL emulation. Adding a custom
 *     scene-graph engine on top would be premature; we use QtQuick3D primitives.
 *   - macOS-native: QQuickWindow uses the Metal graphics API (forced in
 *     `meshroom/ui/app.py:44`); QtQuick3D inherits that. No mode-switching.
 *   - Anti-pattern avoided: no mixing of Qt3D Scene3D and QtQuick3D in the
 *     same QML tree. The legacy Viewer3D remains gated off; MetalScenePreview
 *     is a parallel, opt-in module.
 *
 * Inputs:
 *   - `scenePreviewFolder` (string) — path to a ScenePreview node's output
 *     folder. The viewer parses `<folder>/scene_preview.json` via the
 *     `ScenePreviewLoader` Python helper to discover the renderable model.
 *   - `camerasFile` (string, optional) — if non-empty, overrides the
 *     manifest's cameras path and renders frustums from that SfMData JSON.
 *
 * Outputs (read-only properties for the host UI):
 *   - `loaded` (bool)
 *   - `modelLoaded` (bool)
 *   - `modelStatus` (string)
 */

pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick3D
import QtQuick3D.Helpers     // for OrbitCameraController (Qt 6.4+)
import QtQuick3D.AssetUtils as Loaders   // exposes RuntimeLoader (Qt 6.4+)

import Meshroom.Helpers 1.0   // ScenePreviewLoader, registered in components/__init__.py


Item {
    id: root

    // -------- public API --------
    property alias scenePreviewFolder: previewLoader.scenePreviewFolder
    property alias loaded: previewLoader.isLoaded
    property string camerasFile: ""           // optional override of the manifest's cameras path

    property bool modelLoaded: false
    property string modelStatus: "idle"
    readonly property string modelPath: previewLoader.modelPath
    readonly property string masksPath: previewLoader.masksPath

    // Defensive sizing: when this Item is hosted inside a Component
    // wrapped by a Loader, the natural anchors.fill in our consumers can
    // miss a layout cycle and instantiate us at 0x0. Falling back to the
    // parent's width/height keeps the View3D's Metal render target sized
    // correctly on first paint.
    implicitWidth: parent ? parent.width : 200
    implicitHeight: parent ? parent.height : 150

    signal sceneRefreshed

    // -------- Python-side manifest parser --------
    ScenePreviewLoader {
        id: previewLoader
        onLoaded: {
            // The RuntimeLoader binding auto-refreshes when modelUrl
            // changes. Status transitions are reported by RuntimeLoader's
            // onStatusChanged below; do NOT overwrite root.modelStatus
            // here or we race ahead of the RuntimeLoader's Success report.
            root.sceneRefreshed()
        }
    }

    // -------- the actual 3D view --------
    View3D {
        id: view
        anchors.fill: parent
        // Underlay mode lets QtQuick3D render directly into QtQuick's
        // Metal backing store. The previous Offscreen mode kept the
        // scene in a separate render target whose size sometimes did
        // not match the View3D Item — causing the viewer to appear
        // "small" inside its panel. Underlay tracks the Item's size
        // every frame via the Metal RHI's scissor/viewport, so a
        // resize of the host panel resizes the rendered scene cleanly.
        renderMode: View3D.Underlay
        camera: orbitCamera

        environment: SceneEnvironment {
            antialiasingMode: SceneEnvironment.MSAA
            antialiasingQuality: SceneEnvironment.High
            clearColor: Qt.rgba(0.10, 0.11, 0.12, 1.0)
            backgroundMode: SceneEnvironment.Color
        }

        Node {
            id: sceneRoot

            // World-space camera + light.
            PerspectiveCamera {
                id: orbitCamera
                position: Qt.vector3d(0, 1.2, 3.0)
                eulerRotation.x: -15
                clipNear: 0.05
                clipFar: 5000.0
                fieldOfView: 45.0
            }

            DirectionalLight {
                eulerRotation.x: -45
                eulerRotation.y: 35
                brightness: 1.4
                ambientColor: Qt.rgba(0.25, 0.26, 0.28, 1.0)
            }

            // Ground reference grid — purely visual, swaps out when a model loads.
            Model {
                visible: !root.modelLoaded
                source: "#Rectangle"
                eulerRotation.x: -90
                scale: Qt.vector3d(50, 50, 1)
                materials: PrincipledMaterial {
                    baseColor: Qt.rgba(0.18, 0.19, 0.20, 1.0)
                    roughness: 0.95
                    cullMode: PrincipledMaterial.NoCulling
                }
            }

            // The user's reconstructed mesh — used for GLTF/GLB/OBJ
            // formats QtQuick3D handles natively via RuntimeLoader.
            Loaders.RuntimeLoader {
                id: modelLoader
                visible: previewLoader.isLoaded && previewLoader.modelPath !== ""
                source: previewLoader.modelUrl
                instancing: null
                onStatusChanged: {
                    if (status === Loaders.RuntimeLoader.Success) {
                        root.modelLoaded = true
                        root.modelStatus = "loaded"
                        root.fitCameraToBounds(modelLoader)
                    } else if (status === Loaders.RuntimeLoader.Error) {
                        root.modelLoaded = false
                        root.modelStatus = "error: " + modelLoader.errorString
                        console.warn("[MetalScenePreview] mesh load failed:",
                                     modelLoader.errorString)
                    } else if (status === Loaders.RuntimeLoader.Loading) {
                        root.modelStatus = "loading"
                    }
                }
            }

            // PLY point cloud rendering. RuntimeLoader does not retain
            // per-vertex color from PLY (the canonical format for the
            // dense `densePointCloud.ply` and the SfM `cloud_and_poses.ply`),
            // so we use a custom PointCloudGeometry that parses PLY
            // directly + uploads positions+colors into a single
            // interleaved GPU buffer (Metal RHI shared-storage on macOS).
            // Activates when the manifest reports a non-zero modelFileCount
            // AND no .obj/.gltf was resolved (Python loader returns "" in
            // that case).
            Model {
                id: pointCloud
                visible: previewLoader.isLoaded
                         && previewLoader.modelPath === ""
                         && pointCloudGeo.pointCount > 0
                geometry: PointCloudGeometry {
                    id: pointCloudGeo
                    // The loader's `cameras` symlink points to the SfMData
                    // folder; the dense cloud usually lives next to the
                    // model symlink as `densePointCloud.ply`.
                    source: {
                        if (!previewLoader.isLoaded) return ""
                        // Try the conventional locations.
                        const folder = previewLoader.scenePreviewFolder
                        if (!folder) return ""
                        // Prefer the dense cloud over the SfM cloud.
                        return "file://" + folder + "/model/densePointCloud.ply"
                    }
                    onPointCountChanged: {
                        if (pointCount > 0) {
                            root.modelLoaded = true
                            root.modelStatus = "loaded (" + pointCount + " pts)"
                            root.fitCameraToBounds(pointCloud)
                        }
                    }
                }
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    cullMode: PrincipledMaterial.NoCulling
                }
            }

            // Camera frustums (one per viewpoint).
            CameraFrustumGroup {
                id: frustumGroup
                visible: root.loaded
                sfmDataPath: root.camerasFile !== "" ? root.camerasFile : previewLoader.camerasPath
            }
        }

        // -------- camera controller --------
        // OrbitCameraController gives intuitive tumble+pan+zoom over the
        // origin. Mouse wheel zooms, drag rotates, middle-button pans. Matches
        // the trackball controller the Qt3D viewer used to provide.
        OrbitCameraController {
            anchors.fill: parent
            origin: sceneRoot
            camera: orbitCamera
        }
    }

    function fitCameraToBounds(loadedNode) {
        // Compute a fit that places the orbit camera at 2.5× the bounding-
        // sphere radius from the model centre, looking at the centre.
        // QtQuick3D's loaded Node exposes `.bounds.minimum` and
        // `.bounds.maximum` once the asset finishes; for the PLY point-
        // cloud path we read bounds from the geometry directly.
        var bMin, bMax
        if (loadedNode && loadedNode.bounds) {
            bMin = loadedNode.bounds.minimum
            bMax = loadedNode.bounds.maximum
        } else if (pointCloudGeo && pointCloudGeo.pointCount > 0) {
            // Geometry's setBounds set its own boundsMin/Max — exposed
            // by the geometry. Fall through if not readable.
            bMin = pointCloudGeo.boundsMin
            bMax = pointCloudGeo.boundsMax
        } else {
            bMin = Qt.vector3d(-1, -1, -1)
            bMax = Qt.vector3d(1, 1, 1)
        }
        var cx = (bMin.x + bMax.x) * 0.5
        var cy = (bMin.y + bMax.y) * 0.5
        var cz = (bMin.z + bMax.z) * 0.5
        var dx = bMax.x - bMin.x
        var dy = bMax.y - bMin.y
        var dz = bMax.z - bMin.z
        var radius = 0.5 * Math.sqrt(dx*dx + dy*dy + dz*dz)
        if (radius < 1e-6 || !isFinite(radius)) radius = 1.0
        // Frustum sizing for the frustum group is derived from radius so
        // the per-view markers scale with the scene.
        frustumGroup.frustumNear = Math.max(radius * 0.01, 0.001)
        frustumGroup.frustumFar  = Math.max(radius * 0.04, 0.005)
        orbitCamera.position = Qt.vector3d(cx, cy + 0.15 * radius, cz + 2.5 * radius)
        orbitCamera.lookAt(Qt.vector3d(cx, cy, cz))
        orbitCamera.clipFar = Math.max(orbitCamera.clipFar, radius * 50)
        orbitCamera.clipNear = Math.max(0.01, radius * 0.005)
    }

    // -------- overlay HUD --------
    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.margins: 8
        radius: 4
        color: Qt.rgba(0, 0, 0, 0.55)
        width: hudLabel.width + 16
        height: hudLabel.height + 8
        visible: hudLabel.text.length > 0

        Label {
            id: hudLabel
            anchors.centerIn: parent
            color: "#e0e0e0"
            font.pointSize: 10
            text: {
                if (!previewLoader.isLoaded) return previewLoader.error || "No ScenePreview"
                let parts = []
                if (root.modelLoaded) parts.push("mesh ✓")
                else if (root.modelStatus !== "idle") parts.push("mesh " + root.modelStatus)
                if (previewLoader.masksFileCount > 0) parts.push(previewLoader.masksFileCount + " masks")
                return "Metal · " + parts.join(" · ")
            }
        }
    }
}
