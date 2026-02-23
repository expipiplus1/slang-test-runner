{
  description = "Slang test runner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, crane, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        craneLib = crane.mkLib pkgs;

        slang-test-runner = craneLib.buildPackage {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;

          buildInputs = [ ];
          nativeBuildInputs = [ ];
        };
      in
      {
        packages.default = slang-test-runner;
        packages.slang-test-runner = slang-test-runner;

        apps.default = {
          type = "app";
          program = "${slang-test-runner}/bin/slang-test-runner";
        };
      }
    );
}
