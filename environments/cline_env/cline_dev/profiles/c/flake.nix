{
  description = "C toolchain (C23-capable) for C-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use the default GCC from the pinned nixpkgs; it supports -std=c23.
        cc = pkgs.gcc;

        commonPackages = with pkgs; [
          cc

          cmake
          ninja
          pkg-config

          git
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
            export CC=${cc}/bin/gcc
          '';
        };

        packages.c-env-image = pkgs.dockerTools.buildImage {
          name = "cline-c-env";
          tag = "c23";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "CC=gcc"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-c-env-root";
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

