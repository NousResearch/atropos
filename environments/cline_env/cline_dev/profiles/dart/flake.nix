{
  description = "Dart toolchain for Dart-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        dart = pkgs.dart;

        commonPackages = with pkgs; [
          dart

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
            export PUB_CACHE=$HOME/.pub-cache
            export PATH=$PATH:$PUB_CACHE/bin
          '';
        };

        packages.dart-env-image = pkgs.dockerTools.buildImage {
          name = "cline-dart-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "PUB_CACHE=/root/.pub-cache"
              "PATH=/root/.pub-cache/bin:/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-dart-env-root";
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

