This is a program to obtain the equilibrium configuration of a single ferromagnetic layer by integrating the LLG equation.

How to use:

1) You can execute the 'SingleLayer_SK.sh' script in the terminal.

Check if it is allowed to be executed as a program (you can give a right click --> Properties --> Permissions --> Mark "Allow execution as a program". 

The script will create the output folders, compile the C program 'SingleLayer_SK.c' and execute the program.

2) If you want to plot the configurations (mz component of magnetization), you can execute the python program 'Plot_SingleLayer_SK_Frames.py' by typing 'python Plot_SingleLayer_SK_Frames.py' in the terminal


OBS1: The C program creates files in the folder m_final to store the last configuration just to highlight it.
OBS2: There is a example figure of the skyrmion configuration obtained using the parameters already in the program.
