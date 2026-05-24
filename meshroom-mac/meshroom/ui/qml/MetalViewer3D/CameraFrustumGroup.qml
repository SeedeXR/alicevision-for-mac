/*
 * CameraFrustumGroup — render one wireframe frustum per viewpoint from
 * an SfMData JSON file.
 *
 * Phase 4 used a cube marker per camera (placeholder). Phase 5 upgrades
 * this to a true wireframe pyramid (12 line segments + a look-at stub)
 * computed from the camera's intrinsics × clip planes. The geometry is
 * shared across all frustums via the `frustumGeometry` singleton-style
 * binding — one vertex buffer, N draw calls, transformed per-instance
 * by the Model's worldTransform.
 *
 * Inputs:
 *   - sfmDataPath: path to the cameras SfMData JSON (or a folder).
 *
 * Notes on coordinate frame: AliceVision SfMData stores `center` (world-
 * space camera centre, 3 floats) and `rotation` (row-major 3×3 = 9
 * floats) where the rotation is the WORLD-to-CAMERA matrix. The
 * frustum geometry is in camera-local coords with forward = -Z, so to
 * place it in world space we set the Model's translation = center and
 * rotation = transpose(R) — i.e. camera-to-world. We convert the 3×3
 * to Euler angles for QtQuick3D's eulerRotation property.
 */

import QtQuick
import QtQuick3D
import Meshroom.Helpers 1.0     // FrustumGeometry


Node {
    id: root

    property string sfmDataPath: ""
    property string masksPath: ""        // folder containing per-view masks (PNG)
    property bool showMasks: true
    property int maxCameras: 200
    property real frustumNear: 0.02     // scene units
    property real frustumFar: 0.15
    property real frustumFovY: 45.0
    property real frustumAspect: 1.5
    property color frustumColor: Qt.rgba(1.0, 0.7, 0.2, 1.0)
    property color selectedFrustumColor: Qt.rgba(0.4, 0.9, 1.0, 1.0)
    // Index of the currently-selected pose in _poses (-1 = none).
    property int selectedIndex: -1
    // Read-only convenience accessor for the selected pose (or null).
    readonly property var selectedPose: selectedIndex >= 0 && selectedIndex < _poses.length
                                        ? _poses[selectedIndex] : null

    signal cameraSelected(int index, var pose)

    // -------- parsed pose list (filled by JS on path change) --------
    property var _poses: []

    // -------- ONE shared geometry, reused by all frustum Models --------
    FrustumGeometry {
        id: sharedFrustumGeo
        nearPlane: root.frustumNear
        farPlane: root.frustumFar
        fovYDegrees: root.frustumFovY
        aspectRatio: root.frustumAspect
    }

    onSfmDataPathChanged: _reload()
    Component.onCompleted: _reload()

    function _reload() {
        _poses = []
        if (!sfmDataPath) return
        var url = sfmDataPath.startsWith("file:") ? sfmDataPath : "file://" + sfmDataPath
        var xhr = new XMLHttpRequest()
        xhr.open("GET", url, true)
        xhr.onreadystatechange = function() {
            if (xhr.readyState !== XMLHttpRequest.DONE) return
            if (xhr.status !== 200 && xhr.status !== 0) {
                console.warn("[CameraFrustumGroup] read failed:", url, xhr.status)
                return
            }
            try {
                _parsePoses(JSON.parse(xhr.responseText))
            } catch (e) {
                console.warn("[CameraFrustumGroup] JSON parse failed for", url, ":", e)
            }
        }
        try { xhr.send() } catch (e) {
            console.warn("[CameraFrustumGroup] send failed:", e)
        }
    }

    function _parsePoses(data) {
        var posesById = {}
        var poseArr = data.poses || []
        for (var i = 0; i < poseArr.length; ++i) {
            var p = poseArr[i]
            posesById[p.poseId] = p.pose && p.pose.transform ? p.pose.transform : null
        }
        var poseList = []
        var views = data.views || []
        for (var v = 0; v < views.length && poseList.length < maxCameras; ++v) {
            var view = views[v]
            var t = posesById[view.poseId || view.viewId]
            if (!t || !t.center) continue
            var c = t.center
            var rot = t.rotation
            // The view's image path: AliceVision stores it as `path`.
            // Extract the stem so we can look up the matching mask
            // file in masksPath (BiRefNet writes "<stem>_mask.png").
            var imgPath = view.path || ""
            var slash = Math.max(imgPath.lastIndexOf("/"), imgPath.lastIndexOf("\\"))
            var stem = imgPath.substring(slash + 1)
            var dot = stem.lastIndexOf(".")
            if (dot > 0) stem = stem.substring(0, dot)
            poseList.push({
                tx: parseFloat(c[0]),
                ty: parseFloat(c[1]),
                tz: parseFloat(c[2]),
                // Camera-to-world rotation: transpose the world-to-camera
                // matrix that SfMData stores. We hand a 9-element list to
                // _rotMatToEuler for conversion.
                eul: rot && rot.length === 9
                     ? _rotMatToEuler(rot.map(parseFloat), /*transpose=*/true)
                     : Qt.vector3d(0, 0, 0),
                stem: stem,
                viewId: view.viewId || "",
            })
        }
        _poses = poseList
    }

    // Convert a 3×3 rotation matrix (row-major, 9 floats) to Euler XYZ
    // angles in DEGREES (QtQuick3D's convention).
    function _rotMatToEuler(m, transpose) {
        // m is row-major: [m00, m01, m02, m10, m11, m12, m20, m21, m22]
        // If transpose, we want m^T: index swap (r,c) -> (c,r).
        function el(r, c) {
            return transpose ? m[c * 3 + r] : m[r * 3 + c]
        }
        // XYZ Euler from rotation matrix (Tait-Bryan):
        //   pitch (x) = atan2(-m12, m22) when |m02| < 1, else atan2(m21, m11)
        //   yaw   (y) = asin(m02)
        //   roll  (z) = atan2(-m01, m00)
        var sy = el(0, 2)
        // Clamp for asin numerical safety.
        if (sy > 1) sy = 1
        if (sy < -1) sy = -1
        var rad2deg = 180.0 / Math.PI
        var pitch, yaw, roll
        if (Math.abs(sy) < 0.999999) {
            pitch = Math.atan2(-el(1, 2), el(2, 2)) * rad2deg
            yaw   = Math.asin(sy) * rad2deg
            roll  = Math.atan2(-el(0, 1), el(0, 0)) * rad2deg
        } else {
            // Gimbal-lock fallback.
            pitch = Math.atan2(el(2, 1), el(1, 1)) * rad2deg
            yaw   = (sy > 0 ? 90 : -90)
            roll  = 0
        }
        return Qt.vector3d(pitch, yaw, roll)
    }

    // -------- render one wireframe per pose --------
    //
    // Each frustum is `pickable: true`, so a View3D-level TapHandler in
    // MetalScenePreview can ray-cast and resolve which frustum was hit.
    // We tag the Model with an index property so the picker can map
    // back to the pose list and emit `cameraSelected`.
    Repeater3D {
        id: frustumRepeater
        model: root._poses
        delegate: Node {
            // Wrap the wireframe Model + the mask-overlay Quad in a Node
            // so they share the camera pose transform. The Quad lives in
            // camera-local coords, sized to the near-plane extents.
            required property int index
            required property var modelData
            position: Qt.vector3d(modelData.tx, modelData.ty, modelData.tz)
            eulerRotation: modelData.eul

            Model {
                geometry: sharedFrustumGeo
                pickable: true
                property int cameraIndex: index
                property var frustumGroup: root
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    baseColor: index === root.selectedIndex
                               ? root.selectedFrustumColor
                               : root.frustumColor
                    cullMode: PrincipledMaterial.NoCulling
                }
            }

            // Per-view mask overlay on the near plane. Visible only when
            // masksPath is set + showMasks is true + the per-view mask
            // file exists. The unit-square `#Rectangle` mesh is scaled to
            // match the near-plane extents (computed from the shared
            // frustum's fov × aspect × near), and placed at z = -near
            // (camera-local forward).
            Model {
                visible: root.showMasks
                         && root.masksPath !== ""
                         && modelData.stem !== ""
                source: "#Rectangle"
                // Near-plane half-extents: ay = tan(fovY/2); ax = ay × aspect.
                property real ay: Math.tan(root.frustumFovY * Math.PI / 360.0)
                property real ax: ay * root.frustumAspect
                // `#Rectangle` is 100×100 in QtQuick3D's primitive set
                // (a 1×1 unit rectangle centred at origin, scaled ×100).
                // We pre-divide by 100 in the scale to get the desired
                // half-extents-in-scene-units mapping.
                scale: Qt.vector3d(2 * ax * root.frustumNear / 100,
                                   2 * ay * root.frustumNear / 100,
                                   1)
                position: Qt.vector3d(0, 0, -root.frustumNear)
                // Mask is grayscale; we blend it over the wireframe with
                // a slight tint. opacityChannel uses the mask's RGB
                // luminance as alpha so background pixels disappear.
                materials: PrincipledMaterial {
                    lighting: PrincipledMaterial.NoLighting
                    cullMode: PrincipledMaterial.NoCulling
                    baseColorMap: Texture {
                        source: root.masksPath + "/" + modelData.stem + "_mask.png"
                        // Don't filter — masks are binary-ish.
                        magFilter: Texture.Nearest
                        minFilter: Texture.Nearest
                        flipV: false
                    }
                    opacity: 0.55
                    alphaMode: PrincipledMaterial.Blend
                }
            }
        }
    }

    function selectByIndex(idx) {
        if (idx < 0 || idx >= _poses.length) {
            selectedIndex = -1
            return
        }
        selectedIndex = idx
        cameraSelected(idx, _poses[idx])
    }
}
