@echo off
ECHO Starting Whisper STT Server (Model: large-v3-q5_0)...

:: 절대 경로를 사용하여 서버를 실행.
"C:\whisper.cpp\build\bin\Release\whisper-server.exe" -l ko -m "C:\whisper.cpp\models\ggml-large-v3-q5_0.bin" --host 0.0.0.0 --port 8081

:: 서버가 (오류 등으로) 종료되었을 때 창이 바로 닫히는 것을 방지
ECHO Server has stopped. Press any key to exit.
pause > nul