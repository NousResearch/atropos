{
  description = "C# / .NET toolchain for C# repos";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Use a recent .NET SDK from the pinned nixpkgs.
        dotnet = pkgs.dotnet-sdk_8;

        commonPackages = with pkgs; [
          dotnet

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
            export DOTNET_ROOT=${dotnet}
            export PATH=$DOTNET_ROOT/bin:$PATH
          '';
        };

        packages.csharp-env-image = pkgs.dockerTools.buildImage {
          name = "cline-csharp-env";
          tag = "latest";
          config = {
            Cmd = [ "/bin/bash" ];
            Env = [
              "DOTNET_ROOT=/usr/share/dotnet"
              "PATH=/usr/share/dotnet:/usr/bin:/bin:/opt/bin"
            ];
          };
          copyToRoot = pkgs.buildEnv {
            name = "cline-csharp-env-root";
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

