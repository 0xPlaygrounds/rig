.PHONY: test
test:
	cargo test

.PHONY: test-unit
test-unit:
	cargo test --lib

.PHONY: test-integration
test-integration:
	cargo test --test integration_tests

.PHONY: test-all
test-all:
	cargo test --all

.PHONY: run-example
run-example:
	cargo run --example vector_search_redis

.PHONY: redis-local
redis-local:
	docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

.PHONY: redis-stop
redis-stop:
	docker stop redis-stack && docker rm redis-stack

.PHONY: fix
fix:
	cargo clippy --fix --allow-dirty --allow-staged

.PHONY: fmt
fmt:
	cargo fmt
