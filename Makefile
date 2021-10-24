build: 
	PYTHONPATH=$(pwd) python3 -m build

upload:
	python3 -m twine upload dist/*

clean:
	rm -rf dist 
