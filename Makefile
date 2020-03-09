build:
	@echo "Building Image"
	DOCKER_BUILDKIT=1 docker build -t iclr-cv-model . 

notebook:
	@echo "Starting a Jupyter notebook"
	docker-compose run -p 8889:8888 iclr-cv-model \
		jupyter notebook --ip=0.0.0.0 \
		--NotebookApp.token='' --NotebookApp.password='' \
		--no-browser --allow-root \
		--notebook-dir="notebooks"

run: 
	@echo "Running ICLR model image"
	docker-compose run iclr-cv-model
