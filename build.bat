@REM https://github.com/mstorsjo/llvm-mingw/releases

@SET clang=C:\LLVM\bin\clang
@%clang% -O2 -Wall -Wextra -I include -c performance.c -o performance.o
@%clang% -O2 -Wall -Wextra -I include -c tokenizer.c -o tokenizer.o
@%clang% -O2 -Wall -Wextra -I include -c rwkv4.c -o rwkv4.o
@%clang% -O2 -Wall -Wextra -I include -c main.c -o main.o
@%clang% -O2 -Wall -Wextra -L . performance.o tokenizer.o rwkv4.o main.o -o rwkvonnx.exe -lonnxruntime
@pause
