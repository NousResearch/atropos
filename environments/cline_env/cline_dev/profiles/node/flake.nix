{
  description = "Node.js LTS toolchain for TypeScript/JavaScript repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use the Nixpkgs LTS Node; upstream currently exposes nodejs_22
        # which corresponds to the active LTS line.
        node = pkgs.nodejs_22;

        commonPackages = with pkgs; [
          node
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
            export NODE_ENV=development
          '';
        };

        packages.node-env-image = pkgs.dockerTools.buildImage {
          name = "cline-node-env";
          tag = "lts";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "NODE_ENV=development"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-node-env-root";
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

