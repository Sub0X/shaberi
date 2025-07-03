.PHONY: generate judge filter help

# Usage example:
# make generate m=MODEL_NAME d=DATASET n=NUM_PROC fp=FREQ_PENALTY me=MAX_ENTRIES
# make judge m=MODEL_NAME d=DATASET e=EVALUATOR n=NUM_PROC
# make filter d=DATASET b=BATCH_SIZE

help:
	@echo "Available targets:"
	@echo "  generate  - Generate model answers. Args: m=MODEL_NAME d=DATASET n=NUM_PROC fp=FREQ_PENALTY me=MAX_ENTRIES"
	@echo "  judge     - Judge model answers. Args: m=MODEL_NAME d=DATASET e=EVALUATOR n=NUM_PROC"
	@echo "  filter    - Filter dataset for SFW content. Args: d=DATASET b=BATCH_SIZE"
	@echo "  help      - Show this help message."

generate:
	python ./generate_answers.py $(if $(fp),-fp $(fp)) $(if $(n),-n $(n)) $(if $(m),-m $(m)) $(if $(d),-d $(d)) $(ARGS)

judge:
	python ./judge_answers.py $(if $(fp),-fp $(fp)) $(if $(n),-n $(n)) $(if $(m),-m $(m)) $(if $(d),-d $(d)) $(if $(e),-e $(e)) $(ARGS)

filter:
	python ./filter_dataset.py $(if $(d),-d $(d)) $(if $(b),-b $(b)) $(ARGS)