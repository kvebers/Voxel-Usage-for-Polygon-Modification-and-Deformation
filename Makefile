run:
	python main.py


install:
	pip install -r requirements.txt

setup_env:
	python.exe -m venv venv


activate:
	venv/Scripts/activate