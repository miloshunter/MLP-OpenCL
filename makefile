CONF?=network.conf
EPOCH?=5
export CONF
export EPOCH

all: train run_openmp

train:	
	cd python_training && $(MAKE)

run_openmp:
	cd c_implementation && $(MAKE)


clean:
	@echo "Cleaning up..."
	@rm -f parameters/*.bin
	@cd c_implementation && make clean
