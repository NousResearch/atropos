{
  description = "Swift toolchain for Swift-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use the Swift toolchain from the pinned nixpkgs.
        swift = pkgs.swift;

        commonPackages = with pkgs; [
          swift

          git
          pkg-config
          cmake
          bashInteractive
          gnupg
          which
          coreutils-full
          findutils
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = commonPackages;
          shellHook = ''
            export SWIFT_ENV=development
          '';
        };

        packages.swift-env-image = pkgs.dockerTools.buildImage {
          name = "cline-swift-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "SWIFT_ENV=development"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-swift-env-root";
            paths = commonPackages ++ [
              pkgs.cacert
              pkgs.gitFull
              pkgs.gzip
              pkgs.gnutar
              pkgs.gnumake
            ];
          };
        };
      }
    );
}

