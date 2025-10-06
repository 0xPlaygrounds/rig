{
  description = "Rig development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay
    }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
            inherit system overlays;
          };
      in
      { 
        devShells.default = with pkgs; mkShell {
          buildInputs = [
            pkg-config
            cmake

            openssl
            sqlite
            postgresql
            protobuf

            rust-bin.stable."1.90.0".default
          ];

          OPENSSL_DEV = openssl.dev;
          OPENSSL_LIB_DIR = "${openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${openssl.dev}/include";
        };
      }
    );
}
