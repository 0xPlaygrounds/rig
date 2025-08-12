all: fmt lint

lint:
	cargo clippy
	cargo clippy --tests

build:
	cargo build

release:
	cargo build --bins --release

test:
	cargo test -- --test-threads=1 --nocapture
	cargo test --no-default-features -- --test-threads=1 --nocapture

update:
	cargo update --verbose

fmt:
	cargo fmt
