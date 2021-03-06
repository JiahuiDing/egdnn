CC = g++
#CFLAGS = -g -Wall
CFLAGS = -g -w
CPPFLAGS = -I include -I /usr/include/python3.5m

#all: build egdnn egdnn_python.so
all: egdnn_python.so

egdnn_python.so: src/egdnn_python.cpp src/neuron.cpp src/connection.cpp src/network.cpp src/helper.cpp src/egdnn.cpp
	$(CC) $(CFLAGS) $(CPPFLAGS) -fPIC -shared $^ -o $@ -l python3.5m

egdnn: build/egdnnmain.o build/neuron.o build/connection.o build/network.o build/helper.o build/egdnn.o
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
	
build/egdnn.o: src/egdnn.cpp include/egdnn.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@
	
build:
	mkdir build

clean: 
	rm -rf build/ *~
