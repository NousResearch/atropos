{
  description = "Python 3.10 + Node toolchain for Cline containers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        python = pkgs.python310;
        pythonPackages = pkgs.python310Packages;

        commonPackages = with pkgs; [
          python
          pythonPackages.pip
          pythonPackages.virtualenv

          nodejs_22
          yarn

          git
          pkg-config
          openssl
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
            export PYTHONUNBUFFERED=1
          '';
        };

        packages.python-env-image = pkgs.dockerTools.buildImage {
          name = "cline-python-env";
          tag = "py310";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "PYTHONUNBUFFERED=1"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-python-env-root";
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

