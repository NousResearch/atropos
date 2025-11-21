{
  description = "Ruby toolchain for Ruby-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        ruby = pkgs.ruby_3_3;
        bundler = pkgs.bundler;

        commonPackages = with pkgs; [
          ruby
          bundler

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
            export GEM_HOME=$HOME/.gem/ruby
            export PATH=$GEM_HOME/bin:$PATH
          '';
        };

        packages.ruby-env-image = pkgs.dockerTools.buildImage {
          name = "cline-ruby-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "GEM_HOME=/root/.gem/ruby"
              "PATH=/root/.gem/ruby/bin:/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-ruby-env-root";
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

