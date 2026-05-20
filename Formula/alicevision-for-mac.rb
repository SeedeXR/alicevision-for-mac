# Homebrew formula for the Apple Silicon Metal port of AliceVision.
#
# This formula packages the 12 `aliceVision_*` pipeline binaries plus the
# Metal shader archive (default.metallib) and runtime data (OCIO config,
# LUTs, sensor DB). All twelve binaries are CLIs that compose into the
# AliceVision photogrammetry pipeline:
#
#   cameraInit -> featureExtraction -> imageMatching -> featureMatching
#     -> incrementalSfM -> prepareDenseScene -> depthMapEstimation
#     -> depthMapFiltering -> meshing -> meshFiltering -> texturing
#     (+ importMiddlebury, a Middlebury MVS dataset ingest helper)
#
# Apple Silicon only -- the build hard-errors on x86_64 macOS and on
# non-Apple platforms.

class AliceVisionForMac < Formula
  desc "Apple Silicon Metal port of AliceVision photogrammetry framework"
  homepage "https://github.com/<placeholder>/alicevision-for-mac"
  # Placeholder release URL -- no GitHub release tagged yet. Once the
  # first tag lands, swap this for the real `archive/refs/tags/vX.Y.Z.tar.gz`
  # URL and add the matching `sha256 "..."`.
  url "https://github.com/<placeholder>/alicevision-for-mac/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"
  license "MPL-2.0"
  version "0.1.0"

  # AliceVision-for-mac is Apple Silicon only -- the CMake build aborts on
  # x86_64. Keep this in sync with the `CMAKE_OSX_ARCHITECTURES` guard in
  # CMakeLists.txt.
  depends_on arch: :arm64
  depends_on macos: :sonoma   # matches CMAKE_OSX_DEPLOYMENT_TARGET=14.0
  depends_on xcode: ["15.0", :build]

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "pkgconf" => :build

  # Runtime + link-time deps. All confirmed installed via `brew list`.
  depends_on "alembic"
  depends_on "assimp"
  depends_on "boost"
  depends_on "ceres-solver"
  depends_on "eigen"
  depends_on "geogram"
  depends_on "imath"
  depends_on "lemon"
  depends_on "libomp"
  depends_on "nanoflann"
  depends_on "onnxruntime"
  depends_on "open-mesh"      # Homebrew's OpenMesh package is `open-mesh`
  depends_on "openexr"
  depends_on "openimageio"

  # NOTE: This formula expects the upstream AliceVision source to be
  # included in the release tarball (under upstream/). If the tarball
  # only ships the macOS overlay, add a `resource "upstream"` block
  # pinning the AliceVision commit and unpack it into upstream/ inside
  # `install`.

  def install
    args = %W[
      -G Ninja
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=#{prefix}
      -DAV_BUILD_UPSTREAM=ON
      -DAV_BUILD_UPSTREAM_DEPTHMAP=ON
      -DAV_BUILD_TESTS=OFF
      -DAV_USE_HOMEBREW_DEPS=ON
      -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
      -DCMAKE_OSX_ARCHITECTURES=arm64
    ]

    system "cmake", "-S", ".", "-B", "build", *args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    # `--help` should exit cleanly and print usage. We grep for a stable
    # token ("Usage" or "AliceVision") so we don't depend on the precise
    # wording of upstream's program_options output.
    help_output = shell_output("#{bin}/aliceVision_cameraInit --help")
    assert_match(/AliceVision|Usage/i, help_output)

    # Verify the Metal shader archive landed next to the binaries -- this
    # is what Device::load_library({}) loads via @executable_path lookup.
    assert_predicate bin/"default.metallib", :exist?,
                     "default.metallib must be installed alongside the binaries"

    # Verify the runtime data made it into share/.
    assert_predicate share/"aliceVision/config.ocio", :exist?
    assert_predicate share/"aliceVision/cameraSensors.db", :exist?

    # Sanity: at least one depth-map binary runs end-to-end with --help
    # (the dep tree includes Metal + Ceres + Boost + OIIO, so a clean
    # --help exit covers most of the dynamic-link surface).
    system bin/"aliceVision_depthMapEstimation", "--help"
  end
end
