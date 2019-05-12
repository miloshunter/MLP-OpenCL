CONF?=network.conf
EPOCH?=10
export CONF
export EPOCH

all: train

train:	
	cd python_training && $(MAKE)

generate:
	@echo "Creating empty txt files..."
	touch file-{1..10}.txt
	
clean:
	@echo "Cleaning up..."
	@rm -f parameters/*.bin
	@cd c_implementation && make clean
