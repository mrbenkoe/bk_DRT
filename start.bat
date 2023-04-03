@echo on
call "%SystemDrive%\users\%username%\Miniconda3\Scripts\activate.bat"
cmd /k "activate bk_DRT-Env & jupyter notebook"