{
  description = "Slang test runner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, crane, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (localSystem:
      let
        # Native build
        pkgsNative = import nixpkgs {
          system = localSystem;
          overlays = [ (import rust-overlay) ];
        };

        craneLibNative = crane.mkLib pkgsNative;

        slang-test-runner = craneLibNative.buildPackage {
          src = craneLibNative.cleanCargoSource ./.;
          strictDeps = true;
        };

        # Helper for static Linux musl builds
        mkStaticMuslPackage = { target, linkerEnv }:
          let
            pkgs = import nixpkgs {
              system = localSystem;
              overlays = [ (import rust-overlay) ];
            };

            crossPkgs = import nixpkgs {
              system = localSystem;
              crossSystem.config = "${target}-unknown-linux-musl";
            };

            craneLib = (crane.mkLib pkgs).overrideToolchain (p:
              p.rust-bin.stable.latest.default.override {
                targets = [ "${target}-unknown-linux-musl" ];
              }
            );

            cc = crossPkgs.stdenv.cc;
          in
          craneLib.buildPackage {
            src = craneLib.cleanCargoSource ./.;
            strictDeps = true;

            CARGO_BUILD_TARGET = "${target}-unknown-linux-musl";
            CARGO_BUILD_RUSTFLAGS = "-C target-feature=+crt-static";

            "${linkerEnv}" = "${cc}/bin/${cc.targetPrefix}cc";

            nativeBuildInputs = [ cc ];
            depsBuildBuild = [ pkgs.stdenv.cc ];

            doCheck = false;
          };

        # Helper for x86_64 Windows builds (using mingw)
        mkWindowsX86_64Package =
          let
            pkgs = import nixpkgs {
              system = localSystem;
              overlays = [ (import rust-overlay) ];
              crossSystem = {
                config = "x86_64-w64-mingw32";
              };
            };

            craneLib = (crane.mkLib pkgs).overrideToolchain (p:
              p.rust-bin.stable.latest.default.override {
                targets = [ "x86_64-pc-windows-gnu" ];
              }
            );
          in
          craneLib.buildPackage {
            src = craneLib.cleanCargoSource ./.;
            strictDeps = true;

            CARGO_BUILD_RUSTFLAGS = "-C target-feature=+crt-static";

            doCheck = false;
          };

        # Note: aarch64-windows cross-compilation is not supported due to missing
        # Windows import libraries (kernel32, etc.) in nixpkgs for ARM64.

        # Note: macOS cross-compilation from Linux is not supported by vanilla nixpkgs
        # (Apple's cctools/ld64 are only available on macOS). Build on macOS directly.

        # Linux static builds
        slang-test-runner-x86_64-linux = mkStaticMuslPackage {
          target = "x86_64";
          linkerEnv = "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER";
        };

        slang-test-runner-aarch64-linux = mkStaticMuslPackage {
          target = "aarch64";
          linkerEnv = "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER";
        };

        # Windows build
        slang-test-runner-x86_64-windows = mkWindowsX86_64Package;
      in
      {
        packages = {
          default = slang-test-runner;

          # Cross-compiled static Linux binaries (no external dependencies)
          x86_64-linux-static = slang-test-runner-x86_64-linux;
          aarch64-linux-static = slang-test-runner-aarch64-linux;

          # Cross-compiled Windows binary
          x86_64-windows = slang-test-runner-x86_64-windows;
        };

        apps.default = {
          type = "app";
          program = "${slang-test-runner}/bin/slang-test-runner";
        };

        devShells.default = craneLibNative.devShell {
          packages = [ ];
        };
      }
    );
}
