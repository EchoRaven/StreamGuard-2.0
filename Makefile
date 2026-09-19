PY ?= python3

.PHONY: help install test validate sample synth audit doctor lint clean

help:
	@echo "make install   安装开发依赖"
	@echo "make sample    生成样例 manifest"
	@echo "make test      跑测试"
	@echo "make validate  校验样例(clean 应 0 error, dirty 应 2 error)"
	@echo "make synth     用 lavfi 合成 demo 数据集(无需真实素材)"
	@echo "make audit     压缩域对抗泄漏审计"
	@echo "make doctor    环境自检(驱动/torch/显存/配置/磁盘)"
	@echo "make lint      ruff 检查"

install:
	$(PY) -m pip install -e ".[dev]"

sample:
	$(PY) examples/make_sample.py

test:
	$(PY) -m pytest tests/

validate: sample
	@echo "--- clean (期望 0 error) ---"
	@$(PY) -m sg2.validate examples/manifest.clean.jsonl || true
	@echo "--- dirty (期望 2 error: 拼接泄漏) ---"
	@! $(PY) -m sg2.validate examples/manifest.dirty.jsonl

SYNTH_OUT ?= /tmp/sg2demo

synth:
	$(PY) -m sg2.synth --demo --out $(SYNTH_OUT) --n $(or $(N),12)
	$(PY) -m sg2.validate $(SYNTH_OUT)/manifest.jsonl

audit:
	$(PY) -m sg2.audit $(SYNTH_OUT)/manifest.jsonl $(SYNTH_OUT)/videos

doctor:
	$(PY) scripts/doctor.py --config configs/turing_local.yaml

lint:
	$(PY) -m ruff check sg2/ tests/ examples/

clean:
	rm -rf .pytest_cache __pycache__ sg2/__pycache__ tests/__pycache__ \
	       examples/manifest.*.jsonl *.egg-info
