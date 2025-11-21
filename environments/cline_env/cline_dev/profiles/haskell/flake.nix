{
  description = "Haskell toolchain (GHC) for Haskell-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        ghc = pkgs.haskell.compiler.ghc948 or pkgs.ghc;

        commonPackages = with pkgs; [
          ghc
          cabal-install
          stack

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
            export PATH=$PATH:$HOME/.cabal/bin:$HOME/.local/bin
          '';
        };

        packages.haskell-env-image = pkgs.dockerTools.buildImage {
          name = "cline-haskell-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "PATH=/usr/bin:/bin:/opt/bin:/root/.cabal/bin:/root/.local/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-haskell-env-root";
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

