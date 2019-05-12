CONF?=default.conf
EPOCH?=2
export CONF
export EPOCH

all: train run_test

train:
	mkdir -p parameters
	cd python_training && $(MAKE)

run_test: get_test_image run_openmp

get_test_image:
	mkdir -p test_pics
	wget https://www.dropbox.com/s/enrn8w7zlkmc6c4/4.png
	mv 4.png test_pics/

run_openmp:
	cd c_implementation && $(MAKE)


clean_all:
	@echo "Cleaning up..."
	@rm -rf parameters
	@rm -rf test_pics
	@find ./ -name "*.conf" -not -name "default.conf" -exec rm {} \;
	@cd c_implementation && make clean

clean:
	@cd c_implementation && make clean
