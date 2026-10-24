.PHONY: demo test train score complaints

demo:        ## run the whole pipeline on synthetic demo data
	python src/make_demo_data.py
	python src/train_model.py --input data/demo_financial_text_labeled.csv --tag demo
	python src/score_text.py --input data/demo_financial_text.csv --model models/sentiment_model_demo.joblib
	python src/complaint_analysis.py --input data/demo_complaints.csv

train:       ## train on real Financial PhraseBank data
	python src/train_model.py --input data/financial_phrasebank.csv

score:
	python src/score_text.py --input data/demo_financial_text.csv

complaints:  ## analyse a real CFPB export
	python src/complaint_analysis.py --input data/cfpb_complaints.csv

test:
	pytest -q
