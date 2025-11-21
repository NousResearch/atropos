{
  description = "POSIX shell environment for pure shell / bash tasks";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        commonPackages = with pkgs; [
          bashInteractive
          coreutils-full
          findutils
          gnugrep
          gawk
          gnused
          diffutils
          gzip
          bzip2
          xz
          gnutar
          util-linux
          git
          curl
          wget
          gnumake
          pkg-config
          openssl
          python3
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = commonPackages;
          shellHook = ''
            export SHELL=/bin/bash
          '';
        };

        packages.shell-env-image = pkgs.dockerTools.buildImage {
          name = "cline-shell-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "SHELL=/bin/bash"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-shell-env-root";
            paths = commonPackages ++ [
              pkgs.cacert
              pkgs.gitFull
              pkgs.gnutar
              pkgs.gzip
              pkgs.gnumake
            ];
          };
        };
      }
    );
}
