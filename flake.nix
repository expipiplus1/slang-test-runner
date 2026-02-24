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
          buildInputs = [ ];
          nativeBuildInputs = [ ];
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

        # Helper for aarch64 Windows builds (using LLVM)
        mkWindowsAarch64Package =
          let
            pkgs = import nixpkgs {
              system = localSystem;
              overlays = [ (import rust-overlay) ];
            };

            craneLib = (crane.mkLib pkgs).overrideToolchain (p:
              p.rust-bin.stable.latest.default.override {
                targets = [ "aarch64-pc-windows-gnullvm" ];
              }
            );
          in
          craneLib.buildPackage {
            src = craneLib.cleanCargoSource ./.;
            strictDeps = true;

            CARGO_BUILD_TARGET = "aarch64-pc-windows-gnullvm";
            CARGO_BUILD_RUSTFLAGS = "-C target-feature=+crt-static";

            # Use lld as the linker for LLVM-based target
            CARGO_TARGET_AARCH64_PC_WINDOWS_GNULLVM_LINKER = "${pkgs.llvmPackages.clang}/bin/clang";

            nativeBuildInputs = [ pkgs.llvmPackages.clang pkgs.llvmPackages.lld ];
            depsBuildBuild = [ pkgs.stdenv.cc ];

            doCheck = false;
          };

        # Linux static builds
        slang-test-runner-x86_64-linux = mkStaticMuslPackage {
          target = "x86_64";
          linkerEnv = "CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_LINKER";
        };

        slang-test-runner-aarch64-linux = mkStaticMuslPackage {
          target = "aarch64";
          linkerEnv = "CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER";
        };

        # Windows builds
        slang-test-runner-x86_64-windows = mkWindowsX86_64Package;
        slang-test-runner-aarch64-windows = mkWindowsAarch64Package;
      in
      {
        packages = {
          default = slang-test-runner;
          slang-test-runner = slang-test-runner;

          # Static Linux binaries
          x86_64-linux-static = slang-test-runner-x86_64-linux;
          aarch64-linux-static = slang-test-runner-aarch64-linux;

          # Windows binaries
          x86_64-windows = slang-test-runner-x86_64-windows;
          aarch64-windows = slang-test-runner-aarch64-windows;
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
