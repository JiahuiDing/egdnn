CC = g++
CFLAGS = -g -Wall
CPPFLAGS = -I include

all: build egdnn

egdnn: build/egdnnmain.o build/neuron.o build/connection.o build/network.o build/helper.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $^ -o $@
	
build/egdnnmain.o: src/egdnnmain.cpp include/egdnnmain.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
	
build/neuron.o: src/neuron.cpp include/neuron.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
	
build/connection.o: src/connection.cpp include/connection.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
	
build/network.o: src/network.cpp include/network.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
	
build/helper.o: src/helper.cpp include/helper.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
	
build:
	mkdir build

clean: 
	rm -rf build/ *~
	rm egdnn
