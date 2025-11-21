{
  description = "Elixir (Erlang/OTP) toolchain for Elixir-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        erlang = pkgs.erlang;
        elixir = pkgs.elixir_1_16;
        hex = pkgs.hex;

        commonPackages = with pkgs; [
          erlang
          elixir
          hex

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
            export MIX_HOME=$HOME/.mix
            export HEX_HOME=$HOME/.hex
            export PATH=$PATH:$MIX_HOME/escripts
          '';
        };

        packages.elixir-env-image = pkgs.dockerTools.buildImage {
          name = "cline-elixir-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "MIX_HOME=/root/.mix"
              "HEX_HOME=/root/.hex"
              "PATH=/root/.mix/escripts:/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-elixir-env-root";
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
