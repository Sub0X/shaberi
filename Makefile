.PHONY: generate judge

# Usage example:
# make generate fp=0.5 n=1 m=MODEL_NAME d=DATASET
# make judge fp=0.5 n=1 m=MODEL_NAME d=DATASET e=EVALUATOR

generate:
	python ./generate_answers.py $(if $(fp),-fp $(fp)) $(if $(n),-n $(n)) $(if $(m),-m $(m)) $(if $(d),-d $(d)) $(ARGS)

judge:
	python ./judge_answers.py $(if $(fp),-fp $(fp)) $(if $(n),-n $(n)) $(if $(m),-m $(m)) $(if $(d),-d $(d)) $(if $(e),-e $(e)) $(ARGS)
