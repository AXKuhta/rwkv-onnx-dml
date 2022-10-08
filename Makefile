# NOTE: libonnxruntime.so is a symlink! This makefile will attempt to create it automatically
# libonnxruntime.so.1.12.1 is the real library file
# Take it from onnxruntime-linux-x64-1.12.1.tgz
#
# https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz

OBJS	= performance.o tokenizer.o rwkv4.o main.o libonnxruntime.so
SOURCE	= performance.c tokenizer.c rwkv4.c main.c
HEADER	= 
OUT	= rwkvonnx
CC	 = gcc
FLAGS	 = -g -O2 -c -Wall -I"include/"
LFLAGS	 = '-Wl,-rpath,$$ORIGIN/' -L"." -lonnxruntime -lm

# The '-Wl,-rpath,$$ORIGIN/' adds a special library lookup path into the executable
# https://stackoverflow.com/questions/42344932/how-to-include-correctly-wl-rpath-origin-linker-argument-in-a-makefile

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

performance.o: performance.c
	$(CC) $(FLAGS) performance.c 

tokenizer.o: tokenizer.c
	$(CC) $(FLAGS) tokenizer.c 

rwkv4.o: rwkv4.c
	$(CC) $(FLAGS) rwkv4.c 

main.o: main.c
	$(CC) $(FLAGS) main.c 

libonnxruntime.so:
	ln -s libonnxruntime.so.* libonnxruntime.so

clean:
	rm -f $(OBJS) $(OUT)
