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

train-gpu: 
	@echo "Building GPU Image"
	DOCKER_BUILDKIT=1 docker build -t iclr-cv-model-train . 

	@echo "Running ICLR model train image"
	docker run --runtime nvidia \
		--mount type=bind,source=/home/ubuntu/ICLR-CV-Transfer-Learning,target=/root \
		--mount type=bind,source=/home/ubuntu/ICLR-CV-Transfer-Learning/data,target=/root/data \
		--mount type=bind,source=/home/ubuntu/ICLR-CV-Transfer-Learning/model_weights,target=/root/model_weights \
		iclr-cv-model-train

eval-cpu:
	@echo "Building CPU Image"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile-eval -t iclr-cv-model-eval . 

	@echo "Running ICLR model eval image"
	docker-compose run iclr-cv-model-eval 

eval-gpu: 
	@echo "Building GPU Image"
	DOCKER_BUILDKIT=1 docker build -f Dockerfile-eval-gpu -t iclr-cv-model-eval . 

	@echo "Running ICLR model train image"
	docker run --runtime nvidia \
		--mount type=bind,source=/home/ubuntu/ICLR-CV-Transfer-Learning,target=/root \
		--mount type=bind,source=/home/ubuntu/ICLR-CV-Transfer-Learning/data,target=/root/data \
		--mount type=bind,source=/home/ubuntu/ICLR-CV-Transfer-Learning/model_weights,target=/root/model_weights \
		iclr-cv-model-eval



