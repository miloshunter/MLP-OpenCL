ifndef CONF
override CONF = network.conf
endif
EPOCH?=10


all:
	@echo "Training of a '${CONF}' started"	 
	python3 mnist_train.py ${CONF} ${EPOCH}

clean:
	@rm -rf MNIST_data
