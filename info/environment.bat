@echo on
call "%SystemDrive%\users\%username%\Miniconda3\Scripts\activate.bat"
cmd /k "conda env create -f env.yml"
