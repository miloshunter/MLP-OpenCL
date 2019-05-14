CONF?=default.conf
EPOCH?=2
IMG?=test_pics/4.png
N_THR?=4
export CONF
export EPOCH

all:
	cd c_implementation && $(MAKE)
	cd c_implementation/opencl_implementation && $(MAKE)

train:
	@mkdir -p parameters
	@cd python_training && $(MAKE)

run_test: test_pics/4.png run_single_core run_openmp run_opencl

test_pics/4.png:
	mkdir -p test_pics
	wget https://www.dropbox.com/s/enrn8w7zlkmc6c4/4.png
	mv 4.png test_pics/

run_openmp: c_implementation/simple_mlp.o
	export OMP_NUM_THREADS=$(N_THR) && ./c_implementation/simple_mlp.o $(CONF) $(IMG)

run_opencl: c_implementation/opencl_implementation/opencl_mlp
	./c_implementation/opencl_implementation/opencl_mlp $(CONF) $(IMG)

run_single_core: c_implementation/simple_mlp.o
	export OMP_NUM_THREADS=1 && ./c_implementation/simple_mlp.o $(CONF) $(IMG)

c_implementation/simple_mlp.o:
	cd c_implementation && $(MAKE)

c_implementation/opencl_implementation/opencl_mlp:
	cd c_implementation/opencl_implementation && $(MAKE)



clean_all:
	@echo "Cleaning up..."
	@rm -rf parameters
	@rm -rf test_pics
	@find ./ -name "*.conf" -not -name "default.conf" -exec rm {} \;
	@cd c_implementation && make clean
	@cd python_training && make clean

clean:
	@cd c_implementation && make clean
