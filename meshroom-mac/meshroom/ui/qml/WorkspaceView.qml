import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

import Controls 1.0
import MaterialIcons 2.2
import ImageGallery 1.0
import Viewer 1.0
import Viewer3D 1.0
// Mac-native QtQuick3D viewer (default path; coexists with the legacy
// Qt3D Viewer3D so MESHROOM_ENABLE_VIEWER3D=1 still works for diagnostics).
import MetalViewer3D 1.0

/**
 * WorkspaceView is an aggregation of Meshroom's main modules.
 *
 * It contains an ImageGallery, a 2D and a 3D viewer to manipulate and visualize scene data.
 */

Item {
    id: root

    property variant currentScene: _currentScene
    readonly property variant cameraInits: _currentScene ? _currentScene.cameraInits : null
    property bool readOnly: false
    property alias panel3dViewer: panel3dViewerLoader.item
    property alias panel3dMetalViewer: panel3dMetalViewerLoader.item
    // ScenePreview output folder (set when a ScenePreview node finishes).
    // The Mac-native MetalScenePreview viewer reads scene_preview.json from
    // this path. The host (NodeEditor / scene controller) sets this when
    // the user clicks a ScenePreview node or a run completes.
    property string scenePreviewFolder: ""
    readonly property Viewer2D viewer2D: viewer2D
    readonly property alias imageGallery: imageGallery
    readonly property TextViewer viewerText: textViewer
    property alias mediaViewerTabIndex: mediaViewerPanel.currentTab

    // Text Viewer occupies index 1 when Image Viewer is also visible, else index 0
    readonly property int _textViewerTabIndex: settingsUILayout.showImageViewer ? 1 : 0

    // Use settings instead of visible property as property changes are not propagated
    visible: settingsUILayout.showImageGallery || settingsUILayout.showImageViewer || settingsUILayout.showViewer3D || settingsUILayout.showTextViewer

    // Load a 3D media file in the 3D viewer
    function load3DMedia(filepath, label = undefined) {
        if (panel3dViewerLoader.active) {
            panel3dViewerLoader.item.viewer3D.load(filepath, label)
        }
    }

    Connections {
        target: currentScene
        function onGraphChanged() {
            if (panel3dViewerLoader.active) {
                panel3dViewerLoader.item.viewer3D.clear()
            }
        }
        function onSfmChanged() { viewSfM() }
        function onSfmReportChanged() { viewSfM() }
    }
    Component.onCompleted: viewSfM()

    // Load the current scene's SfM file
    function viewSfM() {
        var activeNode = _currentScene.activeNodes ? _currentScene.activeNodes.get('sfm').node : null
        if (!activeNode)
            return
        if (panel3dViewerLoader.active) {
            panel3dViewerLoader.item.viewer3D.view(activeNode.attribute('output'))
        }
    }

    SystemPalette { id: activePalette }

    MSplitView {
        id: mainSplitView
        anchors.fill: parent
        orientation: Qt.Horizontal

        MSplitView {
            id: leftSplitView
            visible: settingsUILayout.showImageGallery
            orientation: Qt.Vertical
            SplitView.preferredWidth: imageGallery.defaultCellSize * 2 + 20
            SplitView.minimumWidth: imageGallery.defaultCellSize

            ImageGallery {
                id: imageGallery
                visible: settingsUILayout.showImageGallery
                SplitView.fillHeight: true
                readOnly: root.readOnly
                cameraInits: root.cameraInits
                cameraInit: currentScene ? currentScene.cameraInit : null
                tempCameraInit: currentScene ? currentScene.tempCameraInit : null
                cameraInitIndex: currentScene ? currentScene.cameraInitIndex : -1
                onRemoveSelectedImagesRequest: function(objects) { currentScene.removeImages(objects) }
                onAllViewpointsCleared: currentScene.selectedViewId = "-1"
                onFilesDropped: function(drop) {
                    if (drop["meshroomScenes"].length == 1) {
                        ensureSaved(function() {
                            if (currentScene.handleFilesUrl(drop, cameraInit)) {
                                MeshroomApp.addRecentProjectFile(drop["meshroomScenes"][0])
                            }
                        })
                    } else {
                        currentScene.handleFilesUrl(drop, cameraInit)
                    }
                }
            }
        }

        TabPanel {
            id: mediaViewerPanel
            visible: settingsUILayout.showImageViewer || settingsUILayout.showTextViewer
            implicitWidth: Math.round(parent.width * 0.35)
            SplitView.fillWidth: true
            SplitView.minimumWidth: 50

            tabs: {
                var t = []
                if (settingsUILayout.showImageViewer) t.push("Image Viewer")
                if (settingsUILayout.showTextViewer) t.push("Text Viewer")
                return t
            }

            headerBar: RowLayout {
                spacing: 4

                // Loading indicator for image viewer
                BusyIndicator {
                    id: mediaViewerLoadingIndicator
                    padding: 0
                    implicitWidth: 12
                    implicitHeight: 12
                    running: settingsUILayout.showImageViewer && mediaViewerPanel.currentTab === 0 && viewer2D.loadingModules.length > 0
                    visible: running
                }
                Label {
                    visible: mediaViewerLoadingIndicator.visible
                    text: "Loading " + viewer2D.loadingModules
                    font.italic: true
                }

                MaterialToolButton {
                    text: MaterialIcons.more_vert
                    font.pointSize: 11
                    padding: 2
                    checkable: true
                    checked: imageViewerMenu.visible
                    visible: settingsUILayout.showImageViewer && mediaViewerPanel.currentTab === 0
                    onClicked: imageViewerMenu.open()
                    Menu {
                        id: imageViewerMenu
                        y: parent.height
                        x: -width + parent.width
                        Action {
                            id: displayImageToolBarAction
                            text: "Display HDR Toolbar"
                            checkable: true
                            checked: true
                            enabled: viewer2D.useFloatImageViewer
                        }
                        Action {
                            id: displayLensDistortionToolBarAction
                            text: "Display Lens Distortion Toolbar"
                            checkable: true
                            checked: true
                            enabled: viewer2D.useLensDistortionViewer
                        }
                        Action {
                            id: displayPanoramaToolBarAction
                            text: "Display Panorama Toolbar"
                            checkable: true
                            checked: true
                            enabled: viewer2D.usePanoramaViewer
                        }
                        Action {
                            id: displayImagePathAction
                            text: "Display Image Path"
                            checkable: true
                            checked: true && !viewer2D.usePanoramaViewer
                        }
                        Action {
                            id: enable8bitViewerAction
                            text: "Enable 8-bit Viewer"
                            checkable: true
                            checked: MeshroomApp.default8bitViewerEnabled
                        }
                        Action {
                            id: enableSequencePlayerAction
                            text: "Enable Sequence Player"
                            checkable: true
                            checked: MeshroomApp.defaultSequencePlayerEnabled
                        }
                    }
                }
            }

            Viewer2D {
                id: viewer2D
                anchors.fill: parent

                visible: settingsUILayout.showImageViewer && mediaViewerPanel.currentTab === 0

                viewIn3D: root.load3DMedia

                DropArea {
                    anchors.fill: parent
                    keys: ["text/uri-list"]
                    onDropped: function(drop) {
                        viewer2D.loadExternal(drop.urls[0]);
                    }
                }
                Rectangle {
                    z: -1
                    anchors.fill: parent
                    color: Qt.darker(activePalette.base, 1.1)
                }
            }

            TextViewer {
                id: textViewer
                anchors.fill: parent

                visible: settingsUILayout.showTextViewer && mediaViewerPanel.currentTab === root._textViewerTabIndex

                DropArea {
                    anchors.fill: parent
                    keys: ["text/uri-list"]
                    onDropped: function(drop) {
                        textViewer.source = drop.urls[0]
                    }
                }
            }
        }

        Item {
            id: viewer3DContainer
            visible: settingsUILayout.showViewer3D
            Layout.minimumWidth: 20
            Layout.minimumHeight: 80
            Layout.fillHeight: true
            implicitWidth: Math.round(parent.width * 0.45)

            // Mac-native QtQuick3D viewer — the DEFAULT 3D path.
            // Active when:
            //   (a) the user's "Show 3D Viewer" menu toggle is on, AND
            //   (b) _metalViewer3DAvailable is true (set by app.py from
            //       MESHROOM_ENABLE_METAL_VIEWER3D, default "1"), AND
            //   (c) the legacy Qt3D viewer is NOT explicitly enabled —
            //       if a user sets MESHROOM_ENABLE_VIEWER3D=1, they want
            //       the legacy path (probably for diagnostics) and the
            //       metal viewer steps aside. Mutual exclusivity prevents
            //       both viewers from racing for the same canvas.
            Loader {
                id: panel3dMetalViewerLoader
                active: _metalViewer3DAvailable && settingsUILayout.showViewer3D && !_viewer3DAvailable
                visible: active
                anchors.fill: parent
                sourceComponent: panel3dMetalViewerComponent
            }

            Loader {
                id: panel3dViewerLoader
                // Doubly gated: the per-session menu toggle AND the Python-side
                // _viewer3DAvailable context property. The context property is
                // false unless MESHROOM_ENABLE_VIEWER3D=1 is set, preventing the
                // Qt3D Scene3D pipeline-state crash on macOS 26.5 Apple Silicon
                // regardless of what QSettings persisted from a previous run.
                active: _viewer3DAvailable && settingsUILayout.showViewer3D
                visible: active
                anchors.fill: parent
                sourceComponent: panel3dViewerComponent
            }
        }

        Component {
            id: panel3dMetalViewerComponent
            Panel {
                id: panel3dMetalViewer
                title: "3D Viewer (Metal)"
                // Page's contentItem auto-sizes when we don't use a
                // SplitView, but we anchor explicitly to be defensive
                // against macOS Qt 6.11.1 layout-cycle ordering issues.
                anchors.fill: parent

                property alias metalViewer: c_metalViewer3D

                MetalScenePreview {
                    id: c_metalViewer3D
                    anchors.fill: parent
                    // Belt and braces: an explicit Layout.fillWidth/
                    // fillHeight in case a future host wraps this Panel
                    // in a Layout container.
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    // Bind to the most-recently-computed ScenePreview node's
                    // output folder. WorkspaceView's host sets this when the
                    // user selects a ScenePreview node in the graph, or when
                    // a pipeline run finishes.
                    scenePreviewFolder: root.scenePreviewFolder

                    DropArea {
                        anchors.fill: parent
                        keys: ["text/uri-list"]
                        onDropped: function(drop) {
                            if (drop.urls && drop.urls.length > 0) {
                                // Accept a single ScenePreview output folder dragged in.
                                let path = drop.urls[0].toString()
                                if (path.startsWith("file://")) path = path.substring(7)
                                c_metalViewer3D.scenePreviewFolder = path
                            }
                        }
                    }
                }
            }
        }

        Component {
            id: panel3dViewerComponent
            Panel {
                id: panel3dViewer
                title: "3D Viewer"

                property alias viewer3D: c_viewer3D

                MSplitView {
                    id: c_viewer3DSplitView
                    anchors.fill: parent
                    orientation: Qt.Horizontal
                    Viewer3D {
                        id: c_viewer3D

                        SplitView.fillWidth: true
                        SplitView.minimumWidth: 50

                        DropArea {
                            anchors.fill: parent
                            keys: ["text/uri-list"]
                            onDropped: function(drop) {
                                drop.urls.forEach(function(url) {
                                    load3DMedia(url)
                                })
                            }
                        }

                        Connections {
                            target: viewer2D
                            function onSync3DSelectedChanged() {
                                Viewer3DSettings.syncWithPickedViewId = viewer2D.sync3DSelected
                            }
                        }
                    }
                    
                    // Inspector Panel
                    Inspector3D {
                        id: inspector3d
                        SplitView.preferredWidth: 220
                        SplitView.minimumWidth: 100

                        mediaLibrary: c_viewer3D.library
                        camera: c_viewer3D.mainCamera
                        uigraph: currentScene
                        onNodeActivated: _currentScene.setActiveNode(node)
                    }
                }
            }
        }
    }
}
