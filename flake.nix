{
  description = "pylatro development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs:
        let
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              python
              uv
              lua
              stdenv.cc
              gnumake
              pkg-config
            ];

            buildInputs = pkgs.lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ];

            shellHook =
              let
                nixLdLib = "/run/current-system/sw/share/nix-ld/lib";
              in
              ''
                export UV_PYTHON="${python}/bin/python3.12"
                export UV_PYTHON_DOWNLOADS=never
                export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
                export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
                if [ -d "${nixLdLib}" ]; then
                  export LD_LIBRARY_PATH="${nixLdLib}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                fi
              '';
          };
        });
    };
}
