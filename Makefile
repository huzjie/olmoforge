.PHONY: doctor train bench throughput serve test

doctor:
	python -m olmoforge doctor

train:
	python -m olmoforge train

bench:
	python -m olmoforge bench

throughput:
	python -m olmoforge throughput

serve:
	python -m olmoforge serve

test:
	python -m unittest discover -s tests -t .
