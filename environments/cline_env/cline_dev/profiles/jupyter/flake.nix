{
  description = "Python + Jupyter env for notebook-style repos";

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
          pythonPackages.jupyterlab
          pythonPackages.notebook

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
            export JUPYTER_ENABLE_LAB=yes
          '';
        };

        packages.jupyter-env-image = pkgs.dockerTools.buildImage {
          name = "cline-jupyter-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "PYTHONUNBUFFERED=1"
              "JUPYTER_ENABLE_LAB=yes"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-jupyter-env-root";
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

