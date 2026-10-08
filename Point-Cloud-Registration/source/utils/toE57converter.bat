REM Clound compare is  a powerful opensource software tool for managing and manipulating 3D Point Cloud Data
REM The Official web site is here
REM https://www.cloudcompare.org/

REM Cloud Compare has a command line interface that allows for command line access to the software's core features
REM making it useful for automating pount cloud manipulation tasks
REM the command line interface documentation can be found here:
REM https://www.cloudcompare.org/doc/wiki/index.php?title=Command_line_mode
REM the command line interface will only work with the Software is already installed


REM This batch script will convert most supported Cloud Compare file formats to E57
REM This batch script will do two things:
REM 1) subsample the input scan files down to 1 mm and 
REM 2) save resulting subsampled data to E57 file format
REM NOTE: Subsample parameters can be changed by modifying the numeric value that follows the [-SS SPATIAL] parameter 
REM NOTE: the spatial SS value is in meters

REM ==============
REM GENERAL NOTES: 
REM ==============
REM Batch file is called as a windows command
REM batch file is called with 1 parameter for each file to be converted separated by a space character
REM File names need quotes around them in case there are any spaces in path or file name
REM Converted E57 files are placed in the same folder location as the original input file/folder
REM EXAMPLE: run batch file from a windows command prompt
REM [FULLPATHToBATCHFILE] Param1 Param2 Param3 ...


REM ===========================================
REM Sample command call to convert Faro Scans
REM ===========================================
REM Faro Scan conversion will work with either of the following 2 options:
REM FARO OPTION (a) pass a parameter with the actual Faro Scan .fls file  
REM  OR
REM FARO OPTION (b) pass a parameter with the parent folder name (Named the same as the actual .fls file)

REM ===========================================
REM FARO OPTION (a) EXAMPLE
REM ===========================================
REM "c:\BatchFileTest>BlackWolf_iCON.bat" "C:\BatchFileTest\TK_24001.fls\TK_24001.fls" "C:\BatchFileTest\TK_24001.fls\TK_24002.fls" etc

REM ===========================================
REM FARO OPTION (b)
REM ===========================================
REM "c:\BatchFileTest>BlackWolf_iCON.bat" "C:\BatchFileTest\TK_24001.fls" "C:\BatchFileTest\TK_24002.fls" etc


REM ===========================================
REM Sample Command call to Convert PCD files
REM ===========================================
REM "c:\BatchFileTest>BlackWolf_iCON.bat" "C:\BatchFileTest\TK_24001.PCD" "C:\BatchFileTest\TK_24002.PCD" 


REM MAKE SURE THAT THE CORRECT PATH TO CloudCompare.exe is entered below.
REM CloudCompare.exe is installed by default to: "C:\Program Files\CloudCompare\CloudCompare.exe"

for %%f in (%*) DO "C:\Program Files\CloudCompare\CloudCompare.exe" -SILENT -O %%f -C_EXPORT_FMT E57 -NO_TIMESTAMP -SS SPATIAL 0.001
@REM for %%f in (%*) DO "C:\Program Files\CloudCompare\CloudCompare.exe" -SILENT -O %%f -C_EXPORT_FMT E57 -NO_TIMESTAMP -SS SPATIAL 0.001 -LOG_FILE CCSSlog.txt



REM ======================================================================
REM Some details about calling batch files from Python Script
REM NONE OF THE CODE BELOW HAS BEEN TESTED OR VERIFIED
REM ======================================================================

REM ==========================================
REM SIMPLE OPTION
REM import os
REM os.system("run-client.bat param1 param2 param3")
REM ==========================================

REM ==========================================
REM SIMPLE OPTION 2
REM ==========================================
REM import subprocess
REM subprocess.call([r'path where the batch file is stored\name of the batch file.bat'])

REM ======================================================================
REM If you need asynchronous behavior or redirected standard streams.
REM ======================================================================
REM import subprocess
REM from subprocess import *
REM p = Popen(['run-client.bat', param1, param2], stdout=PIPE, stderr=PIPE)
REM output, errors = p.communicate()
REM p.wait() # wait for process to terminate