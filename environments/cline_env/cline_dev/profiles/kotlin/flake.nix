{
  description = "Kotlin (JVM) toolchain for Kotlin-based repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        jdk = pkgs.jdk;
        kotlin = pkgs.kotlin;
        gradle = pkgs.gradle;

        commonPackages = with pkgs; [
          jdk
          kotlin
          gradle

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
            export JAVA_HOME=${jdk}
            export PATH=$JAVA_HOME/bin:$PATH
          '';
        };

        packages.kotlin-env-image = pkgs.dockerTools.buildImage {
          name = "cline-kotlin-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "JAVA_HOME=/usr/lib/jvm"
              "PATH=/usr/lib/jvm/bin:/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-kotlin-env-root";
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

