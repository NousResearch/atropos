{
  description = "Go toolchain for Go-based repos (Cline containers)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use the Go toolchain from the pinned nixpkgs.
        go = pkgs.go;

        commonPackages = with pkgs; [
          go

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
            export GOPATH=$HOME/go
            export PATH=$PATH:$GOPATH/bin
          '';
        };

        packages.go-env-image = pkgs.dockerTools.buildImage {
          name = "cline-go-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "GOPATH=/root/go"
              "PATH=/usr/bin:/bin:/opt/bin:/root/go/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-go-env-root";
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

