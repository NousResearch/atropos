{
  description = "Rust + Node toolchain for Cline containers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        rustToolchain = pkgs.rustup;
        commonPackages = with pkgs; [
          rustToolchain
          nodejs_22
          yarn
          git
          pkg-config
          openssl
          cmake
          python3
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
            export CARGO_NET_GIT_FETCH_WITH_CLI=true
            export PATH=$PATH:$HOME/.cargo/bin
          '';
        };

        packages.rust-env-image = pkgs.dockerTools.buildImage {
          name = "cline-rust-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "CARGO_NET_GIT_FETCH_WITH_CLI=true"
              "PATH=/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-rust-env-root";
            paths = commonPackages ++ [ pkgs.cacert pkgs.gzip pkgs.gnutar pkgs.gitFull pkgs.gnumake ];
          };
        };
      }
    );
}
