#!/usr/bin/env bash
set -xue

if ! [[ "$0" =~ scripts/tests.lint.sh ]]; then
  echo "must be run from repository root"
  exit 255
fi

# https://rust-lang.github.io/rustup/installation/index.html
# rustup toolchain install nightly --allow-downgrade --profile minimal --component clippy
#
# https://github.com/rust-lang/rustfmt
# rustup component add rustfmt
# rustup component add rustfmt --toolchain nightly
# rustup component add clippy
# rustup component add clippy --toolchain nightly

rustup default stable
cargo fmt --all --verbose -- --check

# TODO: enable nightly fmt
rustup default nightly
cargo +nightly fmt --all -- --config-path .rustfmt.nightly.toml --verbose --check || true

# ref. https://github.com/rust-lang/rust-clippy
cargo +nightly clippy --all --all-features -- \
-D clippy::suspicious \
-D clippy::style \
-D clippy::complexity \
-D clippy::perf \
-D clippy::pedantic \
-D clippy::restriction \
-D clippy::nursery \
-D clippy::cargo \
|| true

rustup default stable

echo "ALL SUCCESS!"
