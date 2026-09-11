This story implements a framework for testing this codes functionalities.
Replace the current miscellaneous tests in main.cpp with a proper entry point for calculations (to-be-determined) in the future.
The cppgw executable expects an input file that specifies parameters of the work to be done, and an HDF5 file that contains numerical data to be read in.
The input structure should be ./cppgw -i input_file.txt -d data.hdf5 for a normal calculation
It should also accept ./cppgw input_file.txt data.hdf5 for short.
The input file should be assumed to be encoded as text regardless of the extension.
If the -h flag is passed, print a short help page.
The tests that are currently in src/main.cpp should be moved to their own source file and put in a function.
This function should be called if cppgw is run with the -test or -t flags.
On entry, the program should check whether the inputs are malformed.
Since input file parsing is not implemented in this story, keep the function as a stub for now.
However, if there are any parsing issues with the HDF5 file (with HighFive), throw an exception.

I have placed a sample data file rhf_df.h5 in the build directory. Don't delete it!
I also put a placeholder test.in file that serves as a dummy input file.
This story is considered finished when:
* ./cppgw -i test.in -d rhf_df.h5
* ./cppgw -d rhf_df.h5 -i test.in
* ./cppgw test.in rhf_df.h5
* ./cppgw -h
* ./cppgw -t
* ./cppgw -test
all run without throwing. The first two should check the .h5 file for health, and the last two should run the test suite (-t and -test are aliases).
The test suite invocations are exclusive with the regular run mode: if -i, -d, and -t are specified, that is an invalid invocation.
Invalid invocations should print the help page.
Examples of invalid invocations:
* ./cppgw
* ./cppgw -d foo    
* ./cppgw -i foo
* ./cppgw --baz
