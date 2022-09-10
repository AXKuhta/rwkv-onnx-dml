@REM For some reason TDM-GCC doesn't want to start anywhere except its bin/ folder

@cd C:\TDM-GCC-64\bin
@gcc -O2 -Wall -Wextra -I %~dp0\include -c %~dp0\tokenizer.c -o %~dp0\tokenizer.o
@gcc -O2 -Wall -Wextra -I %~dp0\include -c %~dp0\main.c -o %~dp0\main.o
@gcc -O2 -Wall -Wextra -L %~dp0\ %~dp0\tokenizer.o %~dp0\main.o -o %~dp0\rwkvonnx.exe -lonnxruntime
@pause
