{
  description = "C++ toolchain (C++23-capable) for C++ repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use the default GCC from the pinned nixpkgs; it supports -std=c++23.
        cxx = pkgs.gcc;

        commonPackages = with pkgs; [
          cxx

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
            export CXX=${cxx}/bin/g++
          '';
        };

        packages.cpp-env-image = pkgs.dockerTools.buildImage {
          name = "cline-cpp-env";
          tag = "c++23";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "CXX=g++"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-cpp-env-root";
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

